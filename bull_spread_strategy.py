import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

class BullPutSpreadStrategy:
    def __init__(self, qqq_data, options_data, params):
        """
        Initialize the Bull Put Spread strategy for 0DTE QQQ options
        
        Parameters:
        -----------
        qqq_data : DataFrame
            Historical price data for QQQ
        options_data : DataFrame
            Options chain data for QQQ
        params : dict
            Strategy parameters including:
            - entry_rsi_threshold: RSI level to enter trades
            - profit_target_pct: Target profit as percentage of max profit
            - stop_loss_pct: Stop loss as percentage of max loss
            - width_between_strikes: Width between short and long put
            - max_position_size: Maximum position size as percentage of account
        """
        self.qqq_data = qqq_data
        self.options_data = options_data
        self.params = params
        self.positions = []
        self.trade_history = []
        
        # Add technical indicators
        self.add_indicators()
    
    def add_indicators(self):
        """Add technical indicators to the QQQ data"""
        # Calculate RSI (14-period)
        delta = self.qqq_data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        self.qqq_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Add other indicators as needed (MACD, moving averages, etc.)
        self.qqq_data['sma_20'] = self.qqq_data['close'].rolling(window=20).mean()
        self.qqq_data['sma_50'] = self.qqq_data['close'].rolling(window=50).mean()
    
    def filter_0dte_options(self, date):
        """Filter options data to get only 0DTE options for given date"""
        today_options = self.options_data[self.options_data['date'] == date]
        expiry_today = today_options[today_options['expiration_date'] == date]
        return expiry_today
    
    def find_bull_put_spread(self, date, account_value):
        """
        Find optimal bull put spread for the given date
        Returns short_put, long_put tuples or None if no trade
        """
        # Get current market data
        current_data = self.qqq_data[self.qqq_data['date'] == date].iloc[0]
        current_price = current_data['close']
        current_rsi = current_data['rsi']
        
        # Check entry conditions
        if current_rsi < self.params['entry_rsi_threshold'] or current_data['sma_20'] < current_data['sma_50']:
            return None  # Entry conditions not met
        
        # Get 0DTE options
        todays_options = self.filter_0dte_options(date)
        if todays_options.empty:
            return None  # No 0DTE options available
        
        # Filter puts
        puts = todays_options[todays_options['option_type'] == 'put']
        
        # Find potential short put (around 30 delta)
        short_put_candidates = puts[puts['delta'].between(-0.35, -0.25)]
        if short_put_candidates.empty:
            return None
        
        short_put = short_put_candidates.iloc[0]
        short_strike = short_put['strike']
        
        # Find long put (specified width below short put)
        width = self.params['width_between_strikes']
        long_strike = short_strike - width
        long_put_candidates = puts[puts['strike'] == long_strike]
        
        if long_put_candidates.empty:
            return None
        
        long_put = long_put_candidates.iloc[0]
        
        # Calculate max risk and position size
        max_risk = (short_strike - long_strike) * 100 - (short_put['bid'] - long_put['ask']) * 100
        max_position = account_value * self.params['max_position_size']
        
        # Calculate number of spreads to trade
        num_spreads = int(max_position / max_risk)
        if num_spreads == 0:
            return None
        
        return {
            'short_put': short_put,
            'long_put': long_put,
            'num_spreads': num_spreads,
            'max_risk': max_risk,
            'max_profit': (short_put['bid'] - long_put['ask']) * 100 * num_spreads
        }
    
    def calculate_current_value(self, position, current_options_data):
        """Calculate current value of the position"""
        short_put_current = current_options_data[
            (current_options_data['strike'] == position['short_put']['strike']) & 
            (current_options_data['option_type'] == 'put')
        ]
        
        long_put_current = current_options_data[
            (current_options_data['strike'] == position['long_put']['strike']) & 
            (current_options_data['option_type'] == 'put')
        ]
        
        if short_put_current.empty or long_put_current.empty:
            return None
        
        short_put_value = short_put_current.iloc[0]['ask']
        long_put_value = long_put_current.iloc[0]['bid']
        
        current_value = position['initial_credit'] - (short_put_value - long_put_value) * 100 * position['num_spreads']
        return current_value
    
    def run_backtest(self, start_date, end_date, initial_capital):
        """Run backtest of the strategy"""
        account_value = initial_capital
        current_date = start_date
        results = []
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            # Get current day's data
            day_data = self.qqq_data[self.qqq_data['date'] == current_date]
            if day_data.empty:
                current_date += timedelta(days=1)
                continue
            
            # Check for open positions and manage them
            for position in self.positions:
                current_options = self.filter_0dte_options(current_date)
                if current_options.empty:
                    continue
                
                current_value = self.calculate_current_value(position, current_options)
                if current_value is None:
                    continue
                
                # Check exit conditions
                profit_pct = current_value / position['max_risk']
                
                if profit_pct >= self.params['profit_target_pct'] or \
                   profit_pct <= -self.params['stop_loss_pct'] or \
                   current_date == position['expiry_date']:
                    
                    # Close position
                    account_value += current_value
                    self.trade_history.append({
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'profit_loss': current_value,
                        'profit_pct': profit_pct
                    })
                    self.positions.remove(position)
            
            # Look for new trades
            if not self.positions:  # Only enter if no position exists
                trade = self.find_bull_put_spread(current_date, account_value)
                if trade:
                    initial_credit = (trade['short_put']['bid'] - trade['long_put']['ask']) * 100 * trade['num_spreads']
                    position = {
                        'entry_date': current_date,
                        'expiry_date': current_date,  # For 0DTE, expiry is same day
                        'short_put': trade['short_put'],
                        'long_put': trade['long_put'],
                        'num_spreads': trade['num_spreads'],
                        'initial_credit': initial_credit,
                        'max_risk': trade['max_risk']
                    }
                    self.positions.append(position)
                    
                    # Record state for results
                    results.append({
                        'date': current_date,
                        'account_value': account_value,
                        'action': 'ENTRY',
                        'trade_details': position
                    })
            
            current_date += timedelta(days=1)
        
        # Calculate performance metrics
        if self.trade_history:
            win_trades = sum(1 for trade in self.trade_history if trade['profit_loss'] > 0)
            total_trades = len(self.trade_history)
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            avg_return = np.mean([trade['profit_pct'] for trade in self.trade_history]) if self.trade_history else 0
            
            performance = {
                'total_return': (account_value / initial_capital - 1) * 100,
                'win_rate': win_rate * 100,
                'average_return': avg_return * 100,
                'total_trades': total_trades
            }
        else:
            performance = {
                'total_return': 0,
                'win_rate': 0,
                'average_return': 0,
                'total_trades': 0
            }
        
        return {
            'results': results,
            'trade_history': self.trade_history,
            'performance': performance,
            'final_account': account_value
        }

# Example usage
if __name__ == "__main__":
    # Example parameters
    strategy_params = {
        'entry_rsi_threshold': 40,      # Enter when RSI is below this level (oversold)
        'profit_target_pct': 0.50,      # Exit at 50% of max profit
        'stop_loss_pct': 1.5,           # Exit at 150% of max loss
        'width_between_strikes': 5,     # $5 width between short and long put
        'max_position_size': 0.05       # Use max 5% of account per trade
    }
    
    # You would need to load actual QQQ price data and options data here
    qqq_data = pd.DataFrame()  # Placeholder
    options_data = pd.DataFrame()  # Placeholder
    
    # Initialize and run strategy
    strategy = BullPutSpreadStrategy(qqq_data, options_data, strategy_params)
    
    # Run backtest for 6 months
    start = dt.datetime(2023, 1, 1)
    end = dt.datetime(2023, 6, 30)
    results = strategy.run_backtest(start, end, 100000)
    
    print(f"Final account value: ${results['final_account']:.2f}")
    print(f"Total return: {results['performance']['total_return']:.2f}%")
    print(f"Win rate: {results['performance']['win_rate']:.2f}%")
    print(f"Average return per trade: {results['performance']['average_return']:.2f}%")
    print(f"Total trades: {results['performance']['total_trades']}")