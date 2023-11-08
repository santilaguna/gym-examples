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
# action_space = spaces.Box(low=-K, high=K, shape=(num_dimensions,), dtype=np.int32)
action_space = spaces.MultiDiscrete(
    np.array([2*K + 1 for i in range(num_dimensions)]), 
    dtype=np.int32)  # Nota: debemos restar K a cada acción, dado que el rango debería partir en -K

# default values
dft_stock_symbols = [
    "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DD", "XOM",
    "GE", "GS", "HD", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "PFE", "PG", "TRV", "UNH", "RTX", "VZ", "V", "WMT", "DIS"
]
dft_data_folder = os.path.join("gym-examples", "gym_examples", "envs", "yf_data")

dft_start_date = "2009-01-01"
dft_end_date = "2018-09-30"
dft_balance = 1000000.0
# Create your custom Gym environment with this action space
class YFBasic(gym.Env):
    def __init__(self, stock_symbols=dft_stock_symbols, data_folder=dft_data_folder, start_date=dft_start_date, 
            end_date=dft_end_date, initial_balance=dft_balance):
        super(YFBasic, self).__init__()
        self.action_space = action_space
        self.stock_symbols = stock_symbols
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = np.float32(initial_balance)
        self.env = "train"  # train, val or test
        self.val_init_date = "2015-01-01"
        self.test_init_date = "2016-01-01"
        self.current_step = 0

        # Load historical data for all stock symbols into a dictionary
        self.stock_data = {}
        for symbol in stock_symbols:
            file_path = os.path.join(data_folder, f"{symbol}_historical_data.csv")
            if os.path.exists(file_path):
                self.stock_data[symbol] = pd.read_csv(file_path)  #  index_col=0
            else:
                print(os.listdir())
                print(f"File {file_path} not found")
                raise FileNotFoundError(f"File {file_path} not found")
        
        # Initialize the states space [p, h, b], prices, holdings, balance
        self.observation_space = spaces.Dict({
            "p": spaces.Box(low=0.0, high=np.inf, shape=(num_dimensions,), dtype=np.float32),
            "h": spaces.Box(low=0, high=np.inf, shape=(num_dimensions,), dtype=np.int32),
            "b": spaces.Box(low=0.0, high=np.inf, dtype=np.float32)
            # TODO: add additional information of the state
        })
        self.current_state = {
            "p": self._get_closing_prices(),
            "h": np.zeros(num_dimensions, dtype=np.int32),
            "b": np.array([self.initial_balance], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset your environment (if needed)
        self.current_step = 0
        self.current_state = {
            "p": self._get_closing_prices(),
            "h": np.zeros(num_dimensions, dtype=np.int32),
            "b": np.array([self.initial_balance], dtype=np.float32)
        }
        return self._get_observation(), {}

    def step(self, action):
        # go next trading day to calculate reward
        self.current_step += 1
        # Get the closing prices for the current day
        closing_prices = self._get_closing_prices()
        
        reward, new_state = self._get_reward(closing_prices, action)
        reward /= self.initial_balance

        self.current_state = new_state
        
        data = self.stock_data[self.stock_symbols[0]]  # could be any stock
        if self.env == "train":
            date = data.iloc[self.current_step]["Date"]
            done = date >= self.val_init_date
        elif self.env == "val":
            date = data.iloc[self.current_step]["Date"]
            done = date >= self.test_init_date
        else:
            done = self.current_step > len(data)  # Terminate when data ends
        
        
        if np.isnan(new_state["h"]).any() or np.isnan(new_state["p"]).any() or np.isnan(new_state["b"]):
            print("wololo error")
        if np.isnan(reward):
            print("wololo 2 error")
        return new_state, reward, done, False, {}   # Additional information (if needed)

    def render(self):
        # Render your environment (if needed)
        pass

    def close(self):
        # Clean up resources (if needed)
        pass

    def _get_observation(self):
        return self.current_state
    
    def _get_reward(self, closing_prices, action_):
        # action fix
        action = action_ - K  # np.array([x - K for x in action_], dtype=np.int32)
        portfolio_value = self.current_state["b"][0]  # balance left from previous day
        initial_prices = self.current_state["p"]
        initial_holdings = self.current_state["h"]

        # check if action is posible and initial portfolio value
        balance = self.current_state["b"][0]
        needed_to_buy = 0
        stocks_to_buy = {}
        for i in range(len(self.stock_symbols)):
            portfolio_value += initial_holdings[i] * initial_prices[i]
            # NOTE: sell -> positive in paper
            if action[i] < 0:  # sell
                # if action is greater than holdings, sell all holdings
                if (-1 * action[i]) >= initial_holdings[i]:
                    balance += (initial_holdings[i] * initial_prices[i])
                else:
                    balance += (-1 * action[i]) * initial_prices[i]
            elif action[i] > 0:  # buy
                needed_to_buy += action[i] * initial_prices[i]
                stocks_to_buy[i] = action[i]
            else:  # hold
                pass
        # TODO: cost should be variable, not fixed
        fixed_cost = 1.004  # 0.4% transaction cost (0.2% each way)
        while balance < needed_to_buy * fixed_cost:
            # TODO: find a better way to reduce the number of stocks to buy
            for i in stocks_to_buy:
                if stocks_to_buy[i] <= 0:  # need to keep for later portfolio math
                    continue
                stocks_to_buy[i] -= 1
                needed_to_buy -= initial_prices[i]
                if balance >= needed_to_buy * fixed_cost:
                    break
        needed_to_buy *= fixed_cost
        balance -= needed_to_buy
        # TODO: consider we assume we always can buy at close price, dividends, stock split, etc.
        new_value = 0
        final_holdings = np.zeros(len(self.stock_symbols), dtype=np.int32)
        for i in range(len(self.stock_symbols)):
            if i in stocks_to_buy:
                final_holdings[i] = initial_holdings[i] + stocks_to_buy[i]
            else:
                final_holdings[i] = max(initial_holdings[i] + action[i], 0)
            new_value += final_holdings[i] * closing_prices[i]
        reward = balance + new_value - portfolio_value

        new_state = {
            "p": closing_prices, 
            "h": final_holdings, 
            "b": np.array([balance], dtype=np.float32)
        }
        return reward, new_state
    
    def _get_info(self):
        return {}

    def _get_closing_prices(self):
        closing_prices = []
        for symbol in self.stock_symbols:
            if symbol in self.stock_data:
                data = self.stock_data[symbol]
                if self.current_step < len(data):
                    closing_price = data.iloc[self.current_step]["Close"]
                    closing_prices.append(closing_price)
                else:
                    # check this
                    closing_prices.append(np.float32(0.0))
            else:
                print(self.stock_data.keys())
                print(self.stock_symbols)
                raise Exception(f"Stock data not found for symbol: {symbol}")
                # closing_prices.append(0.0)
        return np.array(closing_prices, dtype=np.float32)

# # Define the stock symbols and date range
# stock_symbols = [
#     "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DD", "XOM",
#     "GE", "GS", "HD", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
#     "NKE", "PFE", "PG", "TRV", "UNH", "RTX", "VZ", "V", "WMT", "DIS"
# ]
# data_folder = os.path.join("gym-examples", "gym_examples", "envs", "yf_data"))

# start_date = "2009-01-01"
# end_date = "2018-09-30"

# Instantiate your custom environment
# env = YFBasic(stock_symbols, data_folder, start_date, end_date)
