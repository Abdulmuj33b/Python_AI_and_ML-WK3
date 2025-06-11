import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gym  # Reinforcement learning environment
from stable_baselines3 import PPO  # RL algorithm

# 1. Fetch Real-Time Transport Data (Example API)
def get_real_time_data():
    url = "https://api.transportdata.com/realtime"  # Replace with actual API
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data['bus_locations'])  # Example structure

# 2. Apply K-Means Clustering for Demand Zones
def cluster_demand(data):
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[['latitude', 'longitude', 'passenger_count']])
    return data

# 3. Reinforcement Learning Environment for Route Optimization
class TransportEnv(gym.Env):
    def __init__(self, data):
        super(TransportEnv, self).__init__()
        self.data = data
        self.state = self.data.sample(1)[['latitude', 'longitude', 'passenger_count']].values
        self.action_space = gym.spaces.Discrete(5)  # 5 possible route adjustments
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def step(self, action):
        reward = -self.data.iloc[action]['passenger_count']  # Minimize congestion
        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.data.sample(1)[['latitude', 'longitude', 'passenger_count']].values
        return self.state

# 4. Train RL Model for Dynamic Route Adjustments
def train_rl_model(data):
    env = TransportEnv(data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

# 5. Execute Optimization
real_time_data = get_real_time_data()
clustered_data = cluster_demand(real_time_data)
rl_model = train_rl_model(clustered_data)

print("Reinforcement Learning Model Trained for Dynamic Route Optimization!")
