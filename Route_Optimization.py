# AI for Sustainable Development (SDG 11: Sustainable Cities)
# Project: Optimizing Public Transport Routes using Machine Learning
# Techniques Used: Real-Time Data Integration, K-Means Clustering, Reinforcement Learning

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gym  # Reinforcement learning framework
from stable_baselines3 import PPO  # Reinforcement learning algorithm

# ===================== STEP 1: FETCH REAL-TIME TRANSPORT DATA ===================== #

def get_real_time_data():
    """
    Fetch real-time public transport data from an API.
    Replace 'API_URL' with a real transport data provider.
    """
    url = "https://api.transportdata.com/realtime"  # Placeholder API URL
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data['bus_locations'])  # Example structure
    else:
        print("Error fetching real-time data!")
        return pd.DataFrame(columns=['latitude', 'longitude', 'passenger_count'])

# ===================== STEP 2: CLUSTER PASSENGER DEMAND ZONES ===================== #

def cluster_demand(data):
    """
    Apply K-Means clustering to group high-demand areas and optimize transport zones.
    """
    if data.empty:
        print("No data available for clustering!")
        return data

    kmeans = KMeans(n_clusters=5, random_state=42)  # Define number of clusters
    data['cluster'] = kmeans.fit_predict(data[['latitude', 'longitude', 'passenger_count']])
    
    # Visualizing the clusters
    plt.scatter(data['longitude'], data['latitude'], c=data['cluster'], cmap='viridis', alpha=0.7)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Optimized Public Transport Demand Zones')
    plt.show()
    
    return data

# ===================== STEP 3: SET UP REINFORCEMENT LEARNING ENVIRONMENT ===================== #

class TransportEnv(gym.Env):
    """
    Custom Reinforcement Learning Environment for optimizing bus routes dynamically.
    """
    def __init__(self, data):
        super(TransportEnv, self).__init__()
        self.data = data
        self.state = self.data.sample(1)[['latitude', 'longitude', 'passenger_count']].values  # Initial state
        self.action_space = gym.spaces.Discrete(5)  # Define 5 possible route adjustments
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # Normalized state space

    def step(self, action):
        """
        Executes a route adjustment and calculates the reward based on passenger congestion.
        Lower passenger count = Better reward.
        """
        reward = -self.data.iloc[action]['passenger_count']  # Minimize congestion
        done = False  # Define stopping condition (for simplicity, training runs indefinitely)
        return self.state, reward, done, {}

    def reset(self):
        """
        Resets the environment and selects a new starting state.
        """
        self.state = self.data.sample(1)[['latitude', 'longitude', 'passenger_count']].values
        return self.state

# ===================== STEP 4: TRAIN REINFORCEMENT LEARNING MODEL ===================== #

def train_rl_model(data):
    """
    Train a reinforcement learning model using Proximal Policy Optimization (PPO)
    to dynamically adjust bus routes based on passenger demand.
    """
    if data.empty:
        print("No data available for RL training!")
        return None

    env = TransportEnv(data)  # Initialize environment
    model = PPO("MlpPolicy", env, verbose=1)  # Define RL algorithm
    model.learn(total_timesteps=10000)  # Train model
    return model

# ===================== STEP 5: EXECUTE TRANSPORT ROUTE OPTIMIZATION ===================== #

# Fetch real-time data
real_time_data = get_real_time_data()

# Apply clustering to group high-demand areas
clustered_data = cluster_demand(real_time_data)

# Train reinforcement learning model for dynamic route optimization
if not clustered_data.empty:
    rl_model = train_rl_model(clustered_data)
    print("✅ Reinforcement Learning Model Trained for Dynamic Route Optimization!")
else:
    print("❌ Reinforcement Learning skipped due to insufficient data.")
