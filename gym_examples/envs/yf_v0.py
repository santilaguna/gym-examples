import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd

# Define the number of dimensions and the percentage range (0% to 100%)
num_dimensions = 30  # TODO: agregar TNX (calcular retorno con respecto a la tasa libre de riesgo)
percentage_range = (0.0, 100.0)

# Create a Box space for the continuous action space
action_space = spaces.Box(
    low=np.array([percentage_range[0]] * num_dimensions),
    high=np.array([percentage_range[1]] * num_dimensions), dtype=float)

# Create your custom Gym environment with this action space
class CustomEnv(gym.Env):
    def __init__(self, stock_symbols, data_folder, start_date, end_date):
        super(CustomEnv, self).__init__()
        self.action_space = action_space
        self.stock_symbols = stock_symbols
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.current_step = 0

        # Load historical data for all stock symbols into a dictionary
        self.stock_data = {}
        for symbol in stock_symbols:
            file_path = os.path.join(data_folder, f"{symbol}_historical_data.csv")
            if os.path.exists(file_path):
                self.stock_data[symbol] = pd.read_csv(file_path, index_col=0)
        
        # Initialize the observation space
        self.observation_space = spaces.Tuple((
            action_space,
            spaces.Box(low=0.0, high=np.inf, shape=(len(stock_symbols),), dtype=np.float32)
        ))

    def reset(self):
        # Reset your environment (if needed)
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        # Enforce the sum of action values to be 100
        normalized_action = action / np.sum(action) * 100
        # Get the closing prices for the current day
        closing_prices = self._get_closing_prices()

        print("Normalized Action:", normalized_action)
        print("Closing Prices:", closing_prices)

        self.current_step += 1

        # Return observations, reward, done, and additional info (if needed)
        observations = (normalized_action, closing_prices)
        reward = 0  # Modify this to calculate the reward
        done = self.current_step > len(self.stock_data[self.stock_symbols[0]])  # Terminate when data ends
        info = {}  # Additional information (if needed)

        return observations, reward, done, info

    def render(self):
        # Render your environment (if needed)
        pass

    def close(self):
        # Clean up resources (if needed)
        pass

    def _get_observation(self):
        return (np.zeros(len(self.stock_symbols)), np.zeros(len(self.stock_symbols)))

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
