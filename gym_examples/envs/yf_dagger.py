from functools import reduce
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd
import math

num_dimensions = 426 #30 426  1, 31

STEP_SIZE = 64 #1
K = 1000  # 1000 if 1 trading day for 0.2% commission

include_all = True
action_type = "weights"  # "continuous", "directions", "multidiscrete", "weights"
if action_type == "continuous":
    # continuous = [-1, 1]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_dimensions,), dtype=np.float32)
elif action_type == "directions":
    # directions = [-1, 0, 1]
    action_space = spaces.MultiDiscrete(
        np.array([3 for i in range(num_dimensions)]), 
        dtype=np.int32)
elif action_type == "multidiscrete":
    # multidiscrete = [-5, ... 0, ...5]
    action_space = spaces.MultiDiscrete(
        np.array([11 for i in range(num_dimensions)]),
        dtype=np.int32)
elif action_type == "weights":
    # weights = [0, 1]
    action_space = spaces.Box(low=0.0, high=1.0, shape=(num_dimensions,), dtype=np.float32)

# dft_stock_symbols = [  #"MMM"]
#     "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DD", "XOM",
#     "GE", "GS", "HD", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
#     "NKE", "PFE", "PG", "TRV", "UNH", "RTX", "VZ", "V", "WMT", "DIS", #"DJI"
# ]

#dft_stock_symbols = ["^GSPC"]
dft_stock_symbols = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'BRK-B', 'GOOG', 'JPM', 'LLY', 'UNH', 'XOM', 'V', 'MA', 'COST', 'HD', 'PG', 'WMT', 'NFLX', 'JNJ', 'CRM', 'BAC', 'ORCL', 'CVX', 'WFC', 'MRK', 'KO', 'CSCO', 'ADBE', 'ACN', 'AMD', 'PEP', 'LIN', 'MCD', 'IBM', 'DIS', 'PM', 'ABT', 'GE', 'CAT', 'TMO', 'ISRG', 'GS', 'VZ', 'TXN', 'INTU', 'QCOM', 'BKNG', 'AXP', 'SPGI', 'T', 'CMCSA', 'MS', 'RTX', 'NEE', 'PGR', 'LOW', 'DHR', 'AMGN', 'ETN', 'HON', 'UNP', 'PFE', 'AMAT', 'BLK', 'TJX', 'COP', 'BX', 'SYK', 'C', 'BSX', 'FI', 'ADP', 'SCHW', 'VRTX', 'TMUS', 'BMY', 'DE', 'MMC', 'SBUX', 'GILD', 'MU', 'LMT', 'BA', 'MDT', 'ADI', 'CB', 'PLD', 'INTC', 'UPS', 'MO', 'SO', 'AMT', 'LRCX', 'TT', 'CI', 'NKE', 'ELV', 'EQIX', 'ICE', 'SHW', 'PH', 'DUK', 'APH', 'MDLZ', 'CMG', 'PNC', 'CDNS', 'KLAC', 'SNPS', 'AON', 'CME', 'USB', 'WM', 'MSI', 'MCK', 'WELL', 'REGN', 'CL', 'MCO', 'CTAS', 'EMR', 'EOG', 'ITW', 'APD', 'CVS', 'COF', 'MMM', 'GD', 'ORLY', 'WMB', 'CSX', 'TDG', 'AJG', 'ADSK', 'FDX', 'MAR', 'NOC', 'OKE', 'BDX', 'TFC', 'ECL', 'NSC', 'FCX', 'SLB', 'PCAR', 'ROP', 'TRV', 'BK', 'DLR', 'SRE', 'TGT', 'FICO', 'URI', 'RCL', 'AFL', 'AMP', 'SPG', 'JCI', 'CPRT', 'PSA', 'ALL', 'GWW', 'AZO', 'AEP', 'CMI', 'MET', 'ROST', 'PWR', 'O', 'D', 'DHI', 'AIG', 'NEM', 'FAST', 'MSCI', 'PEG', 'KMB', 'PAYX', 'LHX', 'FIS', 'CCI', 'PRU', 'PCG', 'DFS', 'AME', 'TEL', 'AXON', 'VLO', 'RSG', 'COR', 'F', 'BKR', 'EW', 'ODFL', 'CBRE', 'LEN', 'DAL', 'HES', 'IT', 'KR', 'CTSH', 'XEL', 'EA', 'EXC', 'A', 'YUM', 'MNST', 'HPQ', 'VMC', 'ACGL', 'SYY', 'GLW', 'MTB', 'KDP', 'RMD', 'GIS', 'MCHP', 'LULU', 'STZ', 'NUE', 'MLM', 'EXR', 'IRM', 'HIG', 'HUM', 'WAB', 'ED', 'DD', 'IDXX', 'NDAQ', 'EIX', 'ROK', 'OXY', 'AVB', 'ETR', 'CSGP', 'GRMN', 'FITB', 'WTW', 'WEC', 'EFX', 'EBAY', 'UAL', 'CNC', 'RJF', 'DXCM', 'TTWO', 'ANSS', 'ON', 'TSCO', 'GPN', 'CAH', 'DECK', 'TPL', 'STT', 'PPG', 'NVR', 'DOV', 'PHM', 'HAL', 'MPWR', 'BR', 'TROW', 'TYL', 'EQT', 'CHD', 'BRO', 'AWK', 'NTAP', 'VTR', 'HBAN', 'EQR', 'MTD', 'DTE', 'PPL', 'ADM', 'CCL', 'HSY', 'AEE', 'RF', 'CINF', 'HUBB', 'SBAC', 'PTC', 'WDC', 'DVN', 'ATO', 'IFF', 'EXPE', 'WY', 'WST', 'WAT', 'BIIB', 'ES', 'WBD', 'ZBH', 'TDY', 'LDOS', 'NTRS', 'PKG', 'K', 'LYV', 'FE', 'BLDR', 'STX', 'STE', 'CNP', 'CMS', 'NRG', 'ZBRA', 'CLX', 'STLD', 'DRI', 'FSLR', 'IP', 'OMC', 'COO', 'LH', 'ESS', 'CTRA', 'MKC', 'SNA', 'WRB', 'LUV', 'MAA', 'BALL', 'PODD', 'FDS', 'PFG', 'HOLX', 'KEY', 'TSN', 'DGX', 'PNR', 'LVS', 'GPC', 'TER', 'TRMB', 'J', 'MAS', 'IEX', 'MOH', 'ARE', 'BBY', 'SMCI', 'ULTA', 'EXPD', 'KIM', 'NI', 'EL', 'BAX', 'GEN', 'EG', 'DPZ', 'AVY', 'LNT', 'ALGN', 'TXT', 'CF', 'L', 'DOC', 'VTRS', 'VRSN', 'JBHT', 'JBL', 'EVRG', 'FFIV', 'POOL', 'ROL', 'RVTY', 'AKAM', 'NDSN', 'TPR', 'DLTR', 'UDR', 'SWK', 'SWKS', 'CPT', 'KMX', 'CAG', 'HST', 'SJM', 'BG', 'JKHY', 'ALB', 'CHRW', 'EMN', 'UHS', 'REG', 'BXP', 'INCY', 'JNPR', 'AIZ', 'TECH', 'IPG', 'ERIE', 'TAP', 'PNW', 'LKQ', 'CRL', 'GL', 'MKTX', 'HSIC', 'HRL', 'CPB', 'TFX', 'RL', 'AES', 'AOS', 'FRT', 'MGM', 'WYNN', 'MTCH', 'HAS', 'APA', 'IVZ', 'MOS', 'CE', 'BWA', 'DVA', 'BF-B', 'FMC', 'MHK', 'BEN', 'PARA', 'WBA']
dft_data_folder = "sp_full_data"  # sp_data dow_data_norm, dow_data_norm_old, paper_data, data/full_data
rf_data_folder = "dow_data_norm"

dft_start_date = "2009-01-01"
dft_end_date = "2015-01-01"
# train start = "2009-01-01"
# val_init_date = "2015-01-01"
# test_init_date = "2016-01-01"
# test end = "2018-09-30"
# test end = "2020-07-06"

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
        self.transaction_cost = 0.002  # 0.2% # TODO: cost should be variable, not fixed

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
        self.rf_data = pd.read_csv(os.path.join(rf_data_folder, "AAPL.csv"))

        if include_all:
            self.not_state_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Date", 
                "rf_daily", "rf_daily_nan"]
        else:
            self.not_state_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Date", 
                "rf_daily", "rf_daily_nan", "h", "b"]  # NOTE: also remove "b" for dummy
        
        self.state_cols = self.stock_data[self.stock_symbols[0]].columns.difference(self.not_state_cols)
        space_dict = {
            col: spaces.Box(low=-4.0, high=4.0, shape=(num_dimensions,), dtype=np.float64) for col in self.state_cols
        }
        # comment for dummy or phb
        nan_cols = [col for col in self.state_cols if "nan" in col]
        for col in nan_cols:
            space_dict[col] = spaces.Box(low=0, high=1, shape=(num_dimensions,), dtype=np.float64)
        # NOTE: comment for dummy or phb
        macro_cols = [col for col in self.state_cols if "rf_change" in col] + ["rf"]
        for col in macro_cols:
            if col in nan_cols:
                space_dict[col] = spaces.Box(low=0, high=1, dtype=np.float64)
            else:
                space_dict[col] = spaces.Box(low=-4.0, high=4.0, dtype=np.float64)

        # space_dict["Close"] = spaces.Box(low=0.0, high=np.inf, shape=(num_dimensions,), dtype=np.float64)
        if include_all:
            space_dict["h"] = spaces.Box(low=0, high=1, shape=(num_dimensions,), dtype=np.float64)
            space_dict["b"] = spaces.Box(low=0, high=1, dtype=np.float64)

            # space_dict["b"] = spaces.Box(low=0, high=2, dtype=np.float64)

        self.observation_space = spaces.Dict(space_dict)

        self.current_pos = 0
        self.rois = []
        self.rfs = []
        self.holdings = [0 for _ in range(num_dimensions)]
        self.invalid_actions = 0
        self.current_sharpe = 0
        self.current_annual_return = 0
        self.reward_type = "roi"  # "roi"
        self.eval = True    # use for evaluation
        self.log = self.eval
        self.current_state = {}
        self.reset()

    def set_dates(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        if self.log:
            print("Dates set to:", start_date, end_date)

    def set_start(self, step):
        self.current_pos += step  # NOTE: it assumes period last more than step

    def set_reward_type(self, type):
        if type in {"roi", "sharpe"}:
            self.reward_type = type

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
        ret = {col: self._get_data(col) for col in self.state_cols}
        ret["h"] = np.zeros(num_dimensions, dtype=np.float64)  # keep
        ret["b"] = np.array([1], dtype=np.float64)  #  keep
        return ret

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
        # Make sure reward is not nan
        if np.isnan(reward):
            reward = 0
        if self.reward_type == "sharpe":
            reward = 0
            if done:
                reward = self.current_sharpe
        return ret_state, reward, done, False, info  # Additional information (if needed)

    def render(self):
        # Render your environment (if needed)
        pass

    def close(self):
        # Clean up resources (if needed)
        pass

    def _get_observation(self):
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
        # FULL STATE
        #ignore = {}
        # SELECT SOME FEATURES
        ignore = {x for x in self.not_state_cols}
        ret_state = {k: v for k, v in self.current_state.items() if k not in ignore}
        # ALL: check there are no nan values
        for k, v in ret_state.items():
            if np.isnan(v).any():  # replace nan with 0
                ret_state[k] = np.nan_to_num(v)
        if include_all:  # need to fix holdings and balance
            # holdings should be weights for the observation
            prices = self._get_data("Close")
            current_holdings = self.current_state["h"]
            balance = self.current_state["b"][0] * self.initial_balance
            portfolio_value = balance
            for i in range(len(self.stock_symbols)):
                portfolio_value += current_holdings[i] * prices[i]
            weights = [h * p / portfolio_value if p > 0 else 0 for h, p in zip(current_holdings, prices)]
            ret_state["h"] = np.array(weights, dtype=np.float64)
            # balance should be normalized with portfolio value as well
            ret_state["b"] = np.array([balance/portfolio_value], dtype=np.float64)
        return ret_state
    
    def _soft_max(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def _action_fix(self, action_):
        if action_type == "continuous":
            # continuous
            return [round(K*x) for x in action_]
        elif action_type == "directions":
            # directions
            # fix = [round(x - 1) for x in action_]
            # return [K*x for x in fix]
            return [round(K*x) for x in action_]
        elif action_type == "multidiscrete":
            # multi_discrete, up to 5
            # fix = [round(x - 5) for x in action_]
            # return [K*x/5 for x in fix]
            return [round(K*x) for x in action_]
        elif action_type == "weights":
            # weights
            new_weights = self._soft_max(action_)
            prices = self._get_data("Close")
            current_holdings = self.current_state["h"]
            portfolio_value = self.current_state["b"][0] * self.initial_balance
            portfolio_value += sum([h * p for h, p in zip(current_holdings, prices)])
            predicted_holdings = [portfolio_value * w // p if p > 0 else 0 for w, p in zip(new_weights, prices)]
            diffs = [p - h for p, h in zip(predicted_holdings, current_holdings)]
            return [max(min(d, K), -K) for d in diffs]
        raise Exception(f"Action type {action_type} not found")

    def _get_reward_and_state(self, action_):
        # action fix
        action = self._action_fix(action_)

        portfolio_value = self.current_state["b"][0] * self.initial_balance  # balance left from previous day
        initial_prices = self._get_data("Close")
        initial_holdings = self.current_state["h"]

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
        new_state = self.get_state_data()
        new_state["h"] = final_holdings
        new_state["b"] = np.array([balance/self.initial_balance], dtype=np.float64)
        if self.eval or self.reward_type == "sharpe":
            return 0, new_state
        return roi, new_state  # train
    
    def get_info(self):
        return self._get_info()

    def _get_data(self, attr, future=0):
        attrs = []
        for symbol in self.stock_symbols:
            if symbol in self.stock_data:
                data = self.stock_data[symbol]
                if self.current_pos <= len(data):
                    if attr in data.iloc[self.current_pos + future]:
                        value = data.iloc[self.current_pos + future][attr]
                        # if pd.isna(value) and self.current_pos == 0:
                        #     value = 0
                        attrs.append(value)
                    else:  # avoid columns with nan, but if needed, fill with 0
                        attrs.append(0)
                    if attr in {"rf", "rf_daily"} or "rf_change" in attr:
                        if attr == "rf_daily":
                            attrs[-1] = self.rf_data.iloc[self.current_pos + future][attr]
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

    def _get_best_action(self):
        # NOTE: seeing 1 step in the future, just to train supervised
        # OPTION 1: sell all bad ones, buy as much as possible of the best ones
        treshold_buy = self.transaction_cost
        treshold_sell = -self.transaction_cost
        max_1stock_percentage = 0.3

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
        # BUYING: proportional to diff
        total_up_diffs = sum([x[1] for x in up_diffs])
        for i, diff in up_diffs:
            stock_cash = new_balance * diff / total_up_diffs
            n = min(K, stock_cash // current_prices[i])
            best_action[i] = n/K

        # adjust to action space, continuous doesn't need this
        if action_type == "continuous":
            return best_action
        elif action_type == "directions":
            # directions = [-1, 0, 1]
            best_action = [round(x) for x in best_action]
            best_action = np.array(best_action, dtype=np.int32)
            return best_action
        elif action_type == "multidiscrete":
            # multi_discrete = [-5, ... 0, ...5]
            best_action = [round(x * 5) for x in best_action]
            best_action = np.array(best_action, dtype=np.int32)
            return best_action
        elif action_type == "weights":
            # weights = [0, 1]
            current_holdings = self.current_state["h"]
            prices = self._get_data("Close")
            ideal_holdings = [current_holdings[i] + best_action[i] for i in range(len(self.stock_symbols))]
            portfolio_value = self.current_state["b"][0] * self.initial_balance
            portfolio_value += sum([h * p for h, p in zip(current_holdings, prices)])
            ideal_weights = [h * p / portfolio_value if p > 0 else 0 for h, p in zip(ideal_holdings, prices)]
            best_action = np.array(ideal_weights, dtype=np.float32)
            return best_action
        else:
            raise Exception(f"Action type {action_type} not found")
    
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
