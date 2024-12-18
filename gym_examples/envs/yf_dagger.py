from functools import reduce
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd
import math

num_dimensions = 30 #30  #1, 31

STEP_SIZE = 64 #1
K = 1000  # 1000 if 1 trading day for 0.2% commission
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
dft_normalize_price = 1  #25  # 25-200
dft_normalize_holdings = 1  #dft_balance / (30 * dft_normalize_price)
# Create your custom Gym environment with this action space
class YFDagger(gym.Env):
    def __init__(self, stock_symbols=dft_stock_symbols, data_folder=dft_data_folder, start_date=dft_start_date, 
            end_date=dft_end_date, initial_balance=dft_balance):
        super(YFDagger, self).__init__()
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

        # Initialize the states
        self.observation_space = spaces.Dict({
            # "Close": spaces.Box(low=0.0, high=np.inf, shape=(num_dimensions,), dtype=np.float64),
            # TODO: test if improves normalizing prices
            "rf": spaces.Box(low=-4.0, high=4.0, dtype=np.float64),
            "rf_change_14": spaces.Box(low=-4.0, high=4.0, dtype=np.float64),
            "rf_change_50": spaces.Box(low=-4.0, high=4.0, dtype=np.float64),
            "rf_change_100": spaces.Box(low=-4.0, high=4.0, dtype=np.float64),
            "MOM_1": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            "MOM_14": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            "VolNorm": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            "VolNorm_nan": spaces.Box(low=0, high=1, shape=(num_dimensions,), dtype=np.float64),
            "OBV_14": spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64),
            # TODO: show if it improves removing holdings and balance from state
            # "h": spaces.Box(low=-1, high=1, shape=(num_dimensions,), dtype=np.float64),
            # "b": spaces.Box(low=0, high=np.inf, dtype=np.float64)
            # "b": spaces.Box(low=0, high=np.inf, dtype=np.float64)

        })
        self.current_pos = 0
        self.rois = []
        self.rfs = []
        self.holdings = [0 for _ in range(num_dimensions)]
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
        self.holdings = [0 for _ in range(num_dimensions)]
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
            "VolNorm": self._get_data("VolNorm") / self.normalize["VolNorm"],  # need nan
            "VolNorm_nan": self._get_data("VolNorm_nan"),
            "OBV_14": self._get_data("OBV_14") / self.normalize["OBV_14"],
            "h": np.zeros(num_dimensions, dtype=np.float64),
            "b": np.array([1], dtype=np.float64)
        }

    def step(self, action):
        reward, new_state = self._get_reward_and_state(action)
        self.current_state = new_state
        
        data = self.stock_data[self.stock_symbols[0]]  # could be any stock
        date = data.iloc[self.current_pos]["Date"]
        done = date >= self.end_date or self.invalid_actions > 5  # 5 invalid actions

        # go next trading day to calculate reward
        self.current_pos += STEP_SIZE
        for i in range(1, STEP_SIZE):  # skip days and mantain valid rewards
            self.rois.append(0)
            self.rfs.append(self._get_data("rf_daily", i)[0])

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
            # reward = annual_return - 1  # NOTE: uncomment to return annual return
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
        # Make sure reward is not nan
        if np.isnan(reward):
            reward = 0
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
        # future_prices = self._get_data("Close", STEP_SIZE)
        # current_prices = self._get_data("Close")
        # # previous_prices = self._get_data("Close", -STEP_SIZE)
        # fixed_up = (1 + self.transaction_cost)
        # fixed_down = (1 - self.transaction_cost)
        # x = []
        # for i in range(len(self.stock_symbols)):
        #     if future_prices[i] * fixed_up > current_prices[i]:
        #         x.append(1.0)
        #     elif future_prices[i] * fixed_down < current_prices[i]:
        #         x.append(-1.0)
        #     else:
        #         x.append(0.0)
        # ULTRA EASY ALTERNATIVES
        # alt 1 proportional to choose best stock?
        # x = [(future_prices[i] - current_prices[i])/current_prices[i] for i in range(len(self.stock_symbols))]
        # return {"h": np.array(x, dtype=np.float64)}
        # SELECT SOME FEATURES
        ret_state = {k: v for k, v in self.current_state.items() if k not in {"h", "b", "Close"}}
        # ALL: check there are no nan values
        for k, v in ret_state.items():
            if np.isnan(v).any():  # replace nan with 0
                ret_state[k] = np.nan_to_num(v)
        return ret_state
    
    def _get_reward_and_state(self, action_):
        # action fix
        action = [round(K*x) for x in action_]

        portfolio_value = self.current_state["b"][0] * self.initial_balance  # balance left from previous day
        initial_prices = self._get_data("Close") * self.normalize_price
        initial_holdings = self.current_state["h"] * self.normalize_holdings

        # check if action is posible and initial portfolio value
        balance = self.current_state["b"][0] * self.initial_balance
        # print("initial balance", balance)
        # print("initial holdings", initial_holdings)
        # print("initial prices", initial_prices)
        # print("prices", self._get_data("Close"))
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
        # print("needed to buy", needed_to_buy * fixed_cost)
        # print("after sell balance", balance)
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
        # print("new balance", balance)
        # TODO: consider we assume we always can buy at close price (add variable slippage), dividends, etc.
        new_value = 0
        final_holdings = np.zeros(len(self.stock_symbols), dtype=np.float64)
                # Get the closing prices for the current day
        closing_prices = self._get_data("Close", STEP_SIZE)
        for i in range(len(self.stock_symbols)):
            if i in stocks_to_buy:
                final_holdings[i] = initial_holdings[i] + stocks_to_buy[i]
            else:
                final_holdings[i] = max(initial_holdings[i] + action[i], 0)
            new_value += final_holdings[i] * closing_prices[i]
        roi = balance + new_value - portfolio_value
        roi /= max(portfolio_value, 1)
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
        return roi, new_state  # NOTE: train roi, new_state / eval 0, new_state
    
    def get_info(self):
        return self._get_info()

    def _get_best_action(self):
        # NOTE: seeing 1 step in the future, just to train supervised
        # OPTION 1: sell all bad ones, buy as much as possible of the best ones
        treshold_buy = self.transaction_cost
        treshold_sell = -self.transaction_cost
        max_1stock_percentage = 0.3
        # OPTION 2: buy if diff > a, sell if diff < b
        # values obtained from supervised random forest results
        # OPTION 3?: buy as much as possible of only the best one?
        # TODO: option 3
        treshold_buy = 1.4889242119790786/100
        treshold_sell = -2.02356011412344/100
        # set best action
        best_action = np.zeros(num_dimensions, dtype=np.float64)
        # See future
        balance = self.current_state["b"][0] * self.initial_balance
        future_prices = self._get_data("Close", min(1, STEP_SIZE))  # TODO: evaluate min days
        current_prices = self._get_data("Close")
        percent_diffs = [(future_prices[i] - current_prices[i])/current_prices[i] for i in range(len(self.stock_symbols))]
        # current holdings percentage
        portfolio_value = balance
        for i in range(len(self.stock_symbols)):
            portfolio_value += self.holdings[i] * current_prices[i]
        proportions = [current_prices[i] * self.holdings[i] / portfolio_value for i in range(len(self.stock_symbols))]
        # adjust tresholds
        if balance/self.initial_balance > 0.5:
            treshold_buy /= 5
        elif balance/self.initial_balance > 0.3:
            treshold_buy /= 3
        elif balance/self.initial_balance > 0.2:
            treshold_buy /= 2

        # SELLING: similar for all options
        # if diff < -fixed_cost, sell
        # if diff > price + fixed_costy buy in order of biggest diff up to K and if enough balance
        up_diffs = []
        new_balance = balance
        for i, diff in enumerate(percent_diffs):
            if diff < treshold_sell:
                best_action[i] = -1
                cashed_stocks = min(K, int(self.holdings[i]))
                cashed_balance = cashed_stocks * current_prices[i] * (1 - self.transaction_cost)
                new_balance += cashed_balance
            elif diff > treshold_buy:
                # filter bigger than max_1stock_percentage
                if proportions[i] > max_1stock_percentage:
                    continue
                up_diffs.append((i, diff))
            # else (TODO: see many steps in the future)
        up_diffs = sorted(up_diffs, key=lambda x: x[1], reverse=True)
        new_balance /= (1 + self.transaction_cost)  # available cash
        # OPTION 1 in order of biggest estimated diff
        # while new_balance > 0 and len(up_diffs) > 0:
        #     i, diff = up_diffs.pop(0)
        #     n = min(K, new_balance // current_prices[i])
        #     best_action[i] = n/K
        #     new_balance -= n * current_prices[i]
        # OPTION 2 proportional to diff
        total_up_diffs = sum([x[1] for x in up_diffs])
        for i, diff in up_diffs:
            stock_cash = new_balance * diff / total_up_diffs
            n = min(K, stock_cash // current_prices[i])
            best_action[i] = n/K
        return best_action
    
    def _get_info(self):
        return {
            "holdings": self.holdings,
            "balance": self.current_state["b"][0] * self.initial_balance,
            "sharpe_ratio": self.current_sharpe,
            "annual_return": self.current_annual_return,
            "pos": self.current_pos,
            "date": self.stock_data[self.stock_symbols[0]].iloc[self.current_pos]["Date"],
            "prices": self._get_data("Close"),
            "best_action": self._get_best_action()
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
