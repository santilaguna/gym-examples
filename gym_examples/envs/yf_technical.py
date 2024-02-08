import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd

num_dimensions = 30  

K = 1000
action_space = spaces.MultiDiscrete(
    np.array([2*K + 1 for i in range(num_dimensions)]), 
    dtype=np.int32)

dft_stock_symbols = [
    "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DD", "XOM",
    "GE", "GS", "HD", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "PFE", "PG", "TRV", "UNH", "RTX", "VZ", "V", "WMT", "DIS"
]
dft_data_folder = "dow_data"
# os.path.join("gym-examples", "gym_examples", "envs", "yf_data")

dft_start_date = "2009-01-01"
dft_end_date = "2015-01-01"
# train start = "2009-01-01"
# val_init_date = "2015-01-01"
# test_init_date = "2016-01-01"
# test end = "2018-09-30"

dft_balance = 1000000.0
dft_normalize_price = 25  # 25-200
dft_normalize_holdings = dft_balance / (30 * dft_normalize_price)
# Create your custom Gym environment with this action space
class YFTechnical(gym.Env):
    def __init__(self, stock_symbols=dft_stock_symbols, data_folder=dft_data_folder, start_date=dft_start_date, 
            end_date=dft_end_date, initial_balance=dft_balance):
        super(YFTechnical, self).__init__()
        self.action_space = action_space
        self.stock_symbols = stock_symbols
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = np.float32(initial_balance)
        self.normalize_price = np.float32(dft_normalize_price)
        self.normalize_holdings = np.float32(dft_normalize_holdings)
        self.transaction_cost = 0.002  # 0.2% # TODO: cost should be variable, not fixed
        self.normalize = {
            "rf": np.float32(0.05),
            "MOM_1": np.float32(0.03),
            "MOM_14": np.float32(0.1),
            "RSI_14_exp": np.float32(50),
            "SHARPE_RATIO": np.float32(2),
            "VolNorm": np.float32(1),
            "OBV_14": np.float32(0.5)
        }

        # Load historical data for all stock symbols into a dictionary
        self.stock_data = {}
        for symbol in stock_symbols:
            file_path = os.path.join(data_folder, f"{symbol}.csv")
            if os.path.exists(file_path):
                self.stock_data[symbol] = pd.read_csv(file_path)
            else:
                print(os.listdir())
                print(f"File {file_path} not found")
                raise FileNotFoundError(f"File {file_path} not found")
        
        self.data_cols = ["Close", "rf", "MOM_1", "MOM_14", "RSI_14_exp", "SHARPE_RATIO", "VolNorm", "OBV_14"]
        # Initialize the states
        self.observation_space = spaces.Dict({
            "Close": spaces.Box(low=0.0, high=np.inf, shape=(num_dimensions,), dtype=np.float32),
            "rf": spaces.Box(low=0, high=np.inf, shape=(num_dimensions,), dtype=np.float32),
            "MOM_1": spaces.Box(low=-1, high=np.inf, shape=(num_dimensions,), dtype=np.float32),
            "MOM_14": spaces.Box(low=-1, high=np.inf, shape=(num_dimensions,), dtype=np.float32),
            "RSI_14_exp": spaces.Box(low=0, high=1, shape=(num_dimensions,), dtype=np.float32),
            "SHARPE_RATIO:": spaces.Box(low=0, high=np.inf, shape=(num_dimensions,), dtype=np.float32),
            "VolNorm": spaces.Box(low=0, high=np.inf, shape=(num_dimensions,), dtype=np.float32),
            "OBV_14": spaces.Box(low=-1, high=1, shape=(num_dimensions,), dtype=np.float32),
            # TODO: test if improves removing holdings and balance from state
            "h": spaces.Box(low=0, high=np.inf, shape=(num_dimensions,), dtype=np.float32),
            "b": spaces.Box(low=0.0, high=np.inf, dtype=np.float32)
        })
        self.current_pos = 0
        self.current_state = {}
        self.reset()

    def set_dates(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Reset your environment (if needed)
        self.current_pos = 0
        data = self.stock_data[self.stock_symbols[0]]  # could be any stock
        date = data.iloc[self.current_pos]["Date"]
        done = date >= self.start_date
        while not done:
            self.current_pos += 1
            date = data.iloc[self.current_pos]["Date"]
            done = date >= self.start_date
        self.current_state = self.get_state_data()
        self.current_state["h"] = np.zeros(num_dimensions, dtype=np.float32)
        self.current_state["b"] = np.array([1], dtype=np.float32)
        return self._get_observation(), {}

    def get_state_data(self):
        return {
            "Close": self._get_data("Close")/self.normalize_price,
            "rf": self._get_data("rf") / self.normalize["rf"],
            "MOM_1": self._get_data("MOM_1") / self.normalize["MOM_1"],
            "MOM_14": self._get_data("MOM_14") / self.normalize["MOM_14"],
            "RSI_14_exp": self._get_data("RSI_14_exp") / self.normalize["RSI_14_exp"],
            "SHARPE_RATIO": self._get_data("SHARPE_RATIO") / self.normalize["SHARPE_RATIO"],
            "VolNorm": self._get_data("VolNorm") / self.normalize["VolNorm"],
            "OBV_14": self._get_data("OBV_14") / self.normalize["OBV_14"],
        }

    def step(self, action):
        # go next trading day to calculate reward
        self.current_pos += 1
        # Get the closing prices for the current day
        closing_prices = self._get_data("Close")
        
        reward, new_state = self._get_reward_and_state(closing_prices, action)

        self.current_state = new_state
        
        data = self.stock_data[self.stock_symbols[0]]  # could be any stock
        date = data.iloc[self.current_pos]["Date"]
        done = date >= self.end_date
        
        if np.isnan(new_state["h"]).any() or np.isnan(new_state["Close"]).any() or np.isnan(new_state["b"]):
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
    
    def _get_reward_and_state(self, closing_prices, action_):
        # action fix
        action = action_ - K  # np.array([x - K for x in action_], dtype=np.int32)
        portfolio_value = self.current_state["b"][0] * self.initial_balance  # balance left from previous day
        initial_prices = self.current_state["Close"] * self.normalize_price
        initial_holdings = self.current_state["h"] * self.normalize_holdings

        # check if action is posible and initial portfolio value
        balance = self.current_state["b"][0] * self.initial_balance
        needed_to_buy = 0
        stocks_to_buy = {}
        fixed_cost = 1 - self.transaction_cost
        for i in range(len(self.stock_symbols)):
            portfolio_value += initial_holdings[i] * initial_prices[i]
            # NOTE: sell -> positive in paper
            if action[i] < 0:  # sell
                # if action is greater than holdings, sell all holdings
                if (-1 * action[i]) >= initial_holdings[i]:
                    balance += (initial_holdings[i] * initial_prices[i]) * fixed_cost
                else:
                    balance += (-1 * action[i]) * initial_prices[i] * fixed_cost
            elif action[i] > 0:  # buy
                needed_to_buy += action[i] * initial_prices[i]
                stocks_to_buy[i] = action[i]
            else:  # hold
                pass
        fixed_cost = 1 + self.transaction_cost
        while balance < needed_to_buy * fixed_cost:
            # TODO: find a better way to reduce the number of stocks to buy
            for i in stocks_to_buy:
                if stocks_to_buy[i] <= 0:  # need to keep for later portfolio math
                    continue
                n = min(stocks_to_buy[i], K // 100)
                stocks_to_buy[i] -= n
                needed_to_buy -= (initial_prices[i] * n)
        needed_to_buy *= fixed_cost
        balance -= needed_to_buy
        # TODO: consider we assume we always can buy at close price (add variable slippage), dividends, etc.
        new_value = 0
        final_holdings = np.zeros(len(self.stock_symbols), dtype=np.float32)
        for i in range(len(self.stock_symbols)):
            if i in stocks_to_buy:
                final_holdings[i] = initial_holdings[i] + stocks_to_buy[i]
            else:
                final_holdings[i] = max(initial_holdings[i] + action[i], 0)
            new_value += final_holdings[i] * closing_prices[i]
        reward = balance + new_value - portfolio_value
        # normalize reward, holdins and prices
        reward /= self.initial_balance
        final_holdings /= self.normalize_holdings
        new_state = self.get_state_data()
        new_state["h"] = final_holdings
        new_state["b"] = np.array([balance/self.initial_balance], dtype=np.float32)
        return reward, new_state
    
    def _get_info(self):
        return {}

    def _get_data(self, attr):
        attrs = []
        for symbol in self.stock_symbols:
            if symbol in self.stock_data:
                data = self.stock_data[symbol]
                if self.current_pos <= len(data):
                    closing_price = data.iloc[self.current_pos][attr]
                    attrs.append(closing_price)
                else:
                    raise Exception(f"Current step {self.current_pos} is greater than data length {len(data)}")
            else:
                print(self.stock_data.keys())
                print(self.stock_symbols)
                raise Exception(f"Stock data not found for symbol: {symbol}")
        return np.array(attrs, dtype=np.float32)
