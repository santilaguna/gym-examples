from functools import reduce
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd
import math

num_dimensions = 30  #1, 31

STEP_SIZE = 1 #1
K = 10000  # 1000 if 1 trading day for 0.2% commission
action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_dimensions,), dtype=np.float32)

dft_stock_symbols = [  #"MMM"]
    "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DD", "XOM",
    "GE", "GS", "HD", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "PFE", "PG", "TRV", "UNH", "RTX", "VZ", "V", "WMT", "DIS", #"DJI"
]
dft_data_folder = "dow_data_norm"
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
class YF30(gym.Env):
    def __init__(self, stock_symbols=dft_stock_symbols, data_folder=dft_data_folder, start_date=dft_start_date, 
            end_date=dft_end_date, initial_balance=dft_balance):
        super(YF30, self).__init__()
        self.action_space = action_space
        self.stock_symbols = stock_symbols
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = np.float64(initial_balance)
        self.normalize_price = np.float64(dft_normalize_price)
        self.normalize_holdings = np.float64(dft_normalize_holdings)
        self.transaction_cost = 0.002  # 0.2% # TODO: cost should be variable, not fixed
        self.normalize = {
            "rf": np.float64(1),
            "MOM_1": np.float64(1),
            "MOM_14": np.float64(1),
            "RSI_14_exp": np.float64(1),
            "SHARPE_RATIO": np.float64(1),
            "VolNorm": np.float64(1),
            "OBV_14": np.float64(1)
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
        
        # original: "Close", "rf", "MOM_1", "MOM_14", "RSI_14_exp", "SHARPE_RATIO", "VolNorm", "OBV_14"
        # self.data_cols = ["Close", "MOM_1", "MOM_14",]
        # Initialize the states
        self.observation_space = spaces.Dict({
            # "Close": spaces.Box(low=0.0, high=np.inf, shape=(num_dimensions,), dtype=np.float64),
            # # TODO: test if improves normalizing prices
            # "rf": spaces.Box(low=-4.0, high=4.0, dtype=np.float64),
            # "rf_change_14": spaces.Box(low=-4.0, high=4.0, dtype=np.float64),
            # "rf_change_50": spaces.Box(low=-4.0, high=4.0, dtype=np.float64),
            # "rf_change_100": spaces.Box(low=-4.0, high=4.0, dtype=np.float64),
            # "MOM_1": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            # "MOM_14": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            # # /#"RSI_14_exp": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            # # /#"SHARPE_RATIO": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            # # /#"SHARPE_RATIO_nan": spaces.Box(low=0, high=1, shape=(num_dimensions,), dtype=np.float64),
            # "VolNorm": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            # "VolNorm_nan": spaces.Box(low=0, high=1, shape=(num_dimensions,), dtype=np.float64),
            # "OBV_14": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            # TODO: test if improves removing holdings and balance from state
            "h": spaces.Box(low=-1, high=1, shape=(num_dimensions,), dtype=np.float64),
            # "b": spaces.Box(low=0, high=np.inf, dtype=np.float64)
        })
        self.current_pos = 0
        self.rois = []
        self.rfs = []
        self.holdings = []
        self.invalid_actions = 0
        self.current_sharpe = 0
        self.current_annual_return = 0
        self.log = False  # use for evaluation
        self.current_state = {}
        self.reset()

    def set_dates(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        if self.log:
            print("Dates set to:", start_date, end_date)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Reset your environment (if needed)
        self.current_pos = 0
        data = self.stock_data[self.stock_symbols[0]]  # could be any stock
        date = data.iloc[self.current_pos]["Date"]
        done = date >= self.start_date
        while not done:
            self.current_pos += STEP_SIZE
            date = data.iloc[self.current_pos]["Date"]
            done = date >= self.start_date
        self.invalid_actions = 0
        self.rois = []
        self.rfs = []
        self.holdings = []
        self.current_state = self.get_state_data()
        return self._get_observation(), {}

    def get_state_data(self):
        return {
            "Close": self._get_data("Close")/self.normalize_price,
            "rf": self._get_data("rf") / self.normalize["rf"],
            "rf_change_14": self._get_data("rf_change_14") / self.normalize["rf"],
            "rf_change_50": self._get_data("rf_change_50") / self.normalize["rf"],
            "rf_change_100": self._get_data("rf_change_100") / self.normalize["rf"],
            "MOM_1": self._get_data("MOM_1") / self.normalize["MOM_1"],
            "MOM_14": self._get_data("MOM_14") / self.normalize["MOM_14"],
            # /#"RSI_14_exp": self._get_data("RSI_14_exp") / self.normalize["RSI_14_exp"],
            # /#"SHARPE_RATIO": self._get_data("SHARPE_RATIO") / self.normalize["SHARPE_RATIO"], #need nan
            # /#"SHARPE_RATIO_nan": self._get_data("SHARPE_RATIO_nan"),
            "VolNorm": self._get_data("VolNorm") / self.normalize["VolNorm"],  # need nan
            "VolNorm_nan": self._get_data("VolNorm_nan"),
            "OBV_14": self._get_data("OBV_14") / self.normalize["OBV_14"],
            "h": np.zeros(num_dimensions, dtype=np.float64),
            "b": np.array([1], dtype=np.float64)
        }

    def step(self, action):
        # go next trading day to calculate reward
        self.current_pos += STEP_SIZE
        for _ in range(STEP_SIZE - 1):  # skip days and mantain valid rewards
            self.rois.append(0)
            self.rfs.append(0)
        # Get the closing prices for the current day
        closing_prices = self._get_data("Close")
        
        reward, new_state = self._get_reward_and_state(closing_prices, action)

        self.current_state = new_state
        
        data = self.stock_data[self.stock_symbols[0]]  # could be any stock
        date = data.iloc[self.current_pos]["Date"]
        done = date >= self.end_date or self.invalid_actions > 5  # 5 invalid actions
        
        # if np.isnan(new_state["h"]).any() or np.isnan(new_state["Close"]).any() or np.isnan(new_state["b"]):
        #     print("wololo error")
        # for col in self.data_cols:
        #     if np.isnan(new_state[col]).any():
        #         print(self.current_state)
        #         print(self.current_pos)
        #         print(date)
        #         print(f"error {col}")
        # if np.isnan(reward):
        #     print("wololo 2 error")
        # calculate total reward
        if done:
            root_power = 252 / len(self.rois)  # 252 trading days and assume len(rois) = len(rfs)
            annual_return = reduce(lambda x, y: x * (1+y), self.rois, 1)
            annual_return = np.power(annual_return, root_power)  # no restar 1
            # NOTE: if we want to return roi return reward here
            total_rf = reduce(lambda x, y: x * (1+y), self.rfs, 1)  # no restar 1
            total_rf = np.power(total_rf, root_power)
            # calculate std 2 years example (sqrt(pitatoria a 504) - sqrt(pitatoria 504 (rf)) / std(pitatoria 504) * sqrt(252)
            std_rois = np.std(self.rois)
            std_rois *= np.sqrt(252)  # annualize std
            # sharpe ratio
            if std_rois == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = (annual_return - total_rf) / std_rois
            self.current_sharpe = sharpe_ratio
            self.current_annual_return = annual_return - 1
            # reward = sharpe_ratio
            reward = annual_return - 1
            if self.log:
                # custom list starts with the date
                custom_list = [str(self.stock_data[self.stock_symbols[0]].iloc[self.current_pos]["Date"])]
                custom_list += [str(self.current_pos), f"SR:{str(self.current_sharpe)}",
                    f"AR:{str(self.current_annual_return)}"]
                custom_str = ",".join(custom_list)
                with open(f"custom_log.txt", "a") as f:
                    f.write(str(custom_str) + "\n")
        info = self._get_info()
        ret_state = self._get_observation()
        return ret_state, reward, done, False, info  # Additional information (if needed)

    def render(self):
        # Render your environment (if needed)
        pass

    def close(self):
        # Clean up resources (if needed)
        pass

    def _get_observation(self):
        # FULL STATE
        # return self.current_state
        # ULTRA BASIC STATE
        # return {"b": np.array([1], dtype=np.float64)}
        # ULTRA EASY STATE
        future_prices = self._get_data("Close", STEP_SIZE)
        current_prices = self._get_data("Close")
        previous_prices = self._get_data("Close", -STEP_SIZE)
        x = [(future_prices[i] > current_prices[i]) for i in range(len(self.stock_symbols))]
        x = [1 if y else -1 for y in x]
        prev_x = [(current_prices[i] > previous_prices[i]) for i in range(len(self.stock_symbols))]
        prev_x = [1 if y else -1 for y in prev_x]
        # ULTRA EASY ALTERNATIVES
        # alt 1, only tops and bottoms
        # aux = []
        # for i in range(len(self.stock_symbols)):
        #     if x[i] == prev_x[i]:  # keep trend no need to buy or sell again
        #         aux.append(0)
        #     else:
        #         aux.append(x[i])
        # alt 2, proportional to choose best stock?
        #x = [(future_prices[i] - current_prices[i])/current_prices[i] for i in range(len(self.stock_symbols))]
        return {"h": np.array(x, dtype=np.float64)}
        # SELECT SOME FEATURES
        # ret_state = {k: v for k, v in self.current_state.items() if k not in {"h", "b", "Close"}}
        # return ret_state
    
    def _get_reward_and_state(self, closing_prices, action_):
        # action fix
        #action = [round(x*K) for x in action_]
        # new action fix
        action = [round(K*x) for x in action_]

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
                n = min(stocks_to_buy[i], K//100)
                stocks_to_buy[i] -= n
                needed_to_buy -= (initial_prices[i] * n)
        # if balance < needed_to_buy * fixed_cost:
        #     self.invalid_actions += 1
        #     self.current_pos -= 1
        #     return -0.1, self.current_state

        needed_to_buy *= fixed_cost
        balance -= needed_to_buy
        # TODO: consider we assume we always can buy at close price (add variable slippage), dividends, etc.
        new_value = 0
        final_holdings = np.zeros(len(self.stock_symbols), dtype=np.float64)
        for i in range(len(self.stock_symbols)):
            if i in stocks_to_buy:
                final_holdings[i] = initial_holdings[i] + stocks_to_buy[i]
            else:
                final_holdings[i] = max(initial_holdings[i] + action[i], 0)
            new_value += final_holdings[i] * closing_prices[i]
        roi = balance + new_value - portfolio_value
        roi /= portfolio_value
        self.rois.append(roi)
        rf = self._get_data("rf_daily")[0]
        self.rfs.append(rf)
        # log info
        self.holdings = [x for x in final_holdings]
        if self.log:
            # custom list starts with the date
            custom_list = [str(self.stock_data[self.stock_symbols[0]].iloc[self.current_pos]["Date"])]
            custom_list += [str(x) for x in final_holdings]
            custom_str = ",".join(custom_list)
            custom_str += f"{str(action_)}-{str(action)}\n"
            with open(f"custom_log.txt", "a") as f:
                f.write(str(custom_str) + "\n")
        # normalize holdings and prices
        final_holdings /= self.normalize_holdings
        new_state = self.get_state_data()
        new_state["h"] = final_holdings
        new_state["b"] = np.array([balance/self.initial_balance], dtype=np.float64)
        return 0, new_state
    
    def _get_info(self):
        return {
            "holdings": self.holdings,
            "sharpe_ratio": self.current_sharpe,
            "annual_return": self.current_annual_return,
            "pos": self.current_pos
        }

    def _get_data(self, attr, future=0):
        attrs = []
        for symbol in self.stock_symbols:
            if symbol in self.stock_data:
                data = self.stock_data[symbol]
                if self.current_pos <= len(data):
                    value = data.iloc[self.current_pos + future][attr]
                    # if pd.isna(value) and self.current_pos == 0:
                    #     value = 0
                    attrs.append(value)
                    if attr in {"rf", "rf_daily"} or "rf_change" in attr:
                        # if pd.isna(value):
                        #     attrs[-1] = 0
                            # print(f"{symbol} {attr} {value} {self.current_pos}")
                        break
                else:
                    raise Exception(f"Current step {self.current_pos} is greater than data length {len(data)}")
            else:
                print(self.stock_data.keys())
                print(self.stock_symbols)
                raise Exception(f"Stock data not found for symbol: {symbol}")
        return np.array(attrs, dtype=np.float64)
    # TODO: USE Adj Close instead of Close
