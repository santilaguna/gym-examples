import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd

# Define the number of dimensions and the percentage range (0% to 100%)
# TODO: agregar TNX (calcular retorno con respecto a la tasa libre de riesgo), no necesariamente como
num_dimensions = 30  
# TODO: use percentage_range for action instead? (0.0, 100.0), use normalized_action 

# Create a Box space for the continuous action space
K = 1000  # max amount of shares to buy
action_space = spaces.Box(low=-K, high=K, shape=(num_dimensions,), dtype=np.int)

# Create your custom Gym environment with this action space
class CustomEnv(gym.Env):
    def __init__(self, stock_symbols, data_folder, start_date, end_date, initial_balance=1000000.0):
        super(CustomEnv, self).__init__()
        self.action_space = action_space
        self.stock_symbols = stock_symbols
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.current_step = 0
        self.initial_balance = initial_balance

        # Load historical data for all stock symbols into a dictionary
        self.stock_data = {}
        for symbol in stock_symbols:
            file_path = os.path.join(data_folder, f"{symbol}_historical_data.csv")
            if os.path.exists(file_path):
                self.stock_data[symbol] = pd.read_csv(file_path, index_col=0)
        
        # Initialize the states space [p, h, b], prices, holdings, balance
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0.0, high=np.inf, shape=(num_dimensions,), dtype=np.float32),
            spaces.Box(low=0, high=np.inf, shape=(num_dimensions,), dtype=np.int),
            gym.spaces.Box(low=0.0, high=np.inf, dtype=np.float32)
            # TODO: add additional information of the state
        ))
        self.current_state = (
            self._get_closing_prices(),
            np.zeros(num_dimensions, dtype=np.int),
            self.initial_balance
        )

    def reset(self):
        # Reset your environment (if needed)
        self.current_step = 0
        self.current_state = (
            self._get_closing_prices(),
            np.zeros(num_dimensions, dtype=np.int),
            self.initial_balance
        )
        return self._get_observation()

    def step(self, action):
        # go next trading day to calculate reward
        self.current_step += 1
        # Get the closing prices for the current day
        closing_prices = self._get_closing_prices()
        
        reward, new_state = self._get_reward(closing_prices, action)
        
        self.current_state = new_state
        
        done = self.current_step > len(self.stock_data[self.stock_symbols[0]])  # Terminate when data ends
        info = {}  # Additional information (if needed)

        return new_state, reward, done, False, info

    def render(self):
        # Render your environment (if needed)
        pass

    def close(self):
        # Clean up resources (if needed)
        pass

    def _get_observation(self):
        return self.current_state
    
    def _get_reward(self, closing_prices, action):
        portfolio_value = self.current_state[2]  # balance left from previous day
        initial_prices = self.current_state[0]
        initial_holdings = self.current_state[1]

        # check if action is posible and initial portfolio value
        balance = self.current_state[2]
        needed_to_buy = 0
        stocks_to_buy = {}
        for i in range(len(self.stock_symbols)):
            portfolio_value += initial_holdings[i] * initial_prices[i]
            # TODO: sell -> negative
            if action[i] > 0:  # sell
                # if action is greater than holdings, sell all holdings
                # NOTE: we are changing action here, it is not a problem but should be careful
                if action[i] > initial_holdings[i]:
                    action[i] = initial_holdings[i]
                balance += action[i] * initial_prices[i]
            elif action[i] < 0:  # buy
                needed_to_buy += -action[i] * initial_prices[i]
                stocks_to_buy[i] = -action[i]
            else:  # hold
                pass
        # TODO: cost should be variable, not fixed
        fixed_cost = 0.996 # 0.4% transaction cost (0.2% each way)
        balance *= fixed_cost
        while balance < needed_to_buy:
            # TODO: find a better way to reduce the number of stocks to buy
            for i in stocks_to_buy:
                stocks_to_buy[i] -= 1
                needed_to_buy -= initial_prices[i]
                if stocks_to_buy[i] == 0:
                    del stocks_to_buy[i]
                if balance >= needed_to_buy:
                    break
        balance -= needed_to_buy
        # TODO: consider we assume we always can buy at close price, dividends, stock split, etc.
        new_value = 0
        final_holdings = np.zeros(len(self.stock_symbols), dtype=np.int)
        for i in range(len(self.stock_symbols)):
            if i in stocks_to_buy:
                final_holdings[i] = initial_holdings[i] + stocks_to_buy[i]
            else:
                final_holdings[i] = initial_holdings[i] - action[i]
            new_value += final_holdings[i] * closing_prices[i]

        reward = balance + new_value - portfolio_value

        new_state = (closing_prices, final_holdings, balance)
        return reward, new_state

    def _get_closing_prices(self):
        closing_prices = []
        for symbol in self.stock_symbols:
            if symbol in self.stock_data:
                data = self.stock_data[symbol]
                if self.current_step < len(data):
                    closing_price = data.iloc[self.current_step]["Close"]
                    closing_prices.append(closing_price)
                else:
                    closing_prices.append(0.0)
            else:
                raise Exception(f"Stock data not found for symbol: {symbol}")
                # closing_prices.append(0.0)
        return np.array(closing_prices, dtype=np.float32)

# Define the stock symbols and date range
stock_symbols = [
    "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DD", "XOM",
    "GE", "GS", "HD", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "PFE", "PG", "TRV", "UNH", "RTX", "VZ", "V", "WMT", "DIS"
]
data_folder = "yf_data"
start_date = "2009-01-01"
end_date = "2018-09-30"

# Instantiate your custom environment
env = CustomEnv(stock_symbols, data_folder, start_date, end_date)
