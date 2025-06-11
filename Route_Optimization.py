# AI for Sustainable Development (SDG 11: Sustainable Cities)
# Project: Optimizing Public Transport Routes using Machine Learning
# Techniques Used: Real-Time Data Integration, K-Means Clustering, Reinforcement Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import folium
from folium.plugins import MarkerCluster
import random
from datetime import datetime, timedelta
import os # For managing output directories

# --- Configuration ---
# Centralized configuration for easy adjustment of parameters
class Config:
    """
    Configuration class for the Public Transport Optimization System.
    Defines parameters for data generation, clustering, RL training, and output.
    """
    # Data Generation Parameters
    CITY_CENTER = (40.7128, -74.0060) # New York City (approx)
    NUM_BUSES_INITIAL = 50
    NUM_STOPS_INITIAL = 200
    SIMULATION_HOUR = 8 # Hour for initial data generation (e.g., 8 AM for morning rush)

    # Clustering Parameters
    N_CLUSTERS = 5 # Default number of clusters for K-Means
    KMEANS_N_INIT = 10 # Number of times K-Means will be run with different centroid seeds

    # Reinforcement Learning Parameters
    RL_TIMESTEPS = 20000 # Total timesteps for RL model training
    RL_LEARNING_RATE = 0.0003
    RL_N_STEPS = 2048 # Number of steps to run for each environment per update
    RL_BATCH_SIZE = 64
    RL_N_EPOCHS = 10
    RL_GAMMA = 0.99
    RL_GAE_LAMBDA = 0.95
    RL_CLIP_RANGE = 0.2
    RL_ENT_COEF = 0.01
    RL_CHECK_FREQ = 1000 # Frequency for logging training progress
    MAX_EPISODE_STEPS = 100 # Maximum steps per episode in the RL environment

    # Output Paths
    OUTPUT_DIR = "transport_optimization_results"
    DEMAND_CLUSTERS_PLOT = os.path.join(OUTPUT_DIR, 'demand_clusters.png')
    TRAINING_PROGRESS_PLOT = os.path.join(OUTPUT_DIR, 'training_progress.png')
    INITIAL_ROUTES_MAP = os.path.join(OUTPUT_DIR, 'initial_routes.html')
    OPTIMIZED_ROUTES_MAP = os.path.join(OUTPUT_DIR, 'optimized_routes.html')
    RL_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "transport_optimization_model")

# ===================== STEP 1: SIMULATED DATA GENERATION ===================== #

class TransportDataGenerator:
    """
    Generates realistic public transport simulation data including bus stops,
    routes, and real-time bus positions with passenger counts.
    """
    def __init__(self, city_center=Config.CITY_CENTER, num_buses=Config.NUM_BUSES_INITIAL,
                 num_stops=Config.NUM_STOPS_INITIAL):
        self.city_center = city_center
        self.num_buses = num_buses # Note: num_buses here is more of a suggestion for real-time gen
        self.num_stops = num_stops
        self.stops = self.generate_stops()
        self.routes = self.generate_routes()

    def generate_stops(self) -> pd.DataFrame:
        """
        Generates bus stops with a realistic distribution, clustered around
        simulated key locations (city center, financial, residential, etc.)
        Each stop also has a base demand.
        """
        stops_data = []
        # Define clusters for stop generation with their properties
        clusters = [
            {'center': self.city_center, 'spread': 0.05, 'count': 30, 'type': 'city_center'},
            {'center': (40.73, -74.03), 'spread': 0.03, 'count': 20, 'type': 'financial'},
            {'center': (40.75, -73.98), 'spread': 0.04, 'count': 25, 'type': 'residential'},
            {'center': (40.70, -74.00), 'spread': 0.03, 'count': 15, 'type': 'university'},
            {'center': (40.68, -73.97), 'spread': 0.05, 'count': 20, 'type': 'industrial'}
        ]

        # Base demand ranges for different area types
        demand_ranges = {
            'city_center': (50, 200),
            'financial': (30, 150),
            'residential': (20, 100),
            'university': (40, 180),
            'industrial': (10, 80)
        }

        for cluster in clusters:
            for _ in range(cluster['count']):
                lat = cluster['center'][0] + random.uniform(-cluster['spread'], cluster['spread'])
                lon = cluster['center'][1] + random.uniform(-cluster['spread'], cluster['spread'])
                min_demand, max_demand = demand_ranges[cluster['type']]
                base_demand = random.randint(min_demand, max_demand)

                stops_data.append({
                    'stop_id': len(stops_data) + 1,
                    'latitude': lat,
                    'longitude': lon,
                    'base_demand': base_demand,
                    'area_type': cluster['type']
                })
        return pd.DataFrame(stops_data)

    def generate_routes(self) -> pd.DataFrame:
        """
        Generates predefined bus routes connecting different area types using
        the generated bus stops. Routes have a frequency and capacity.
        """
        routes_data = []

        # Get indices of stops by area type for more logical routing
        # Use .copy() to avoid SettingWithCopyWarning if these are later modified
        city_center_stops = self.stops[self.stops['area_type'] == 'city_center'].index.tolist()
        financial_stops = self.stops[self.stops['area_type'] == 'financial'].index.tolist()
        residential_stops = self.stops[self.stops['area_type'] == 'residential'].index.tolist()
        university_stops = self.stops[self.stops['area_type'] == 'university'].index.tolist()
        industrial_stops = self.stops[self.stops['area_type'] == 'industrial'].index.tolist()

        # Ensure enough stops exist in each area for route generation
        min_stops_per_area = 3
        if any(len(s) < min_stops_per_area for s in [city_center_stops, financial_stops,
                                                   residential_stops, university_stops,
                                                   industrial_stops]):
            raise ValueError(f"Not enough stops in one or more area types (need at least {min_stops_per_area}) to generate routes. "
                             "Consider increasing num_stops or adjusting cluster counts.")

        # Major route patterns using actual stop IDs from our generated data
        # Each pattern is a sequence of stop_ids
        patterns = [
            # City center to residential
            [
                self.stops.iloc[city_center_stops[0]]['stop_id'],
                self.stops.iloc[city_center_stops[1]]['stop_id'],
                self.stops.iloc[residential_stops[0]]['stop_id'],
                self.stops.iloc[residential_stops[1]]['stop_id']
            ],
            # Financial to industrial
            [
                self.stops.iloc[financial_stops[0]]['stop_id'],
                self.stops.iloc[financial_stops[1]]['stop_id'],
                self.stops.iloc[industrial_stops[0]]['stop_id'],
                self.stops.iloc[industrial_stops[1]]['stop_id']
            ],
            # University loop
            [
                self.stops.iloc[university_stops[0]]['stop_id'],
                self.stops.iloc[university_stops[1]]['stop_id'],
                self.stops.iloc[university_stops[2]]['stop_id'],
                self.stops.iloc[university_stops[0]]['stop_id']  # Loop back
            ],
            # Cross-city express
            [
                self.stops.iloc[city_center_stops[2]]['stop_id'],
                self.stops.iloc[financial_stops[2]]['stop_id'],
                self.stops.iloc[industrial_stops[2]]['stop_id']
            ]
        ]

        for i, pattern in enumerate(patterns):
            route = {
                'route_id': i + 1,
                'stops': pattern,
                'frequency': random.choice([10, 15, 20]),  # minutes between buses
                'capacity': random.randint(40, 80)
            }
            routes_data.append(route)

        return pd.DataFrame(routes_data)

    def generate_real_time_data(self, hour: int = None) -> pd.DataFrame:
        """
        Generates realistic real-time transport data, including bus positions
        and current passenger counts, influenced by temporal demand patterns.
        """
        if hour is None:
            hour = datetime.now().hour

        # Time-based demand multipliers to simulate rush hours, daytime, night
        time_multipliers = {
            'morning_rush': {'start': 7, 'end': 9, 'multiplier': 2.0},
            'evening_rush': {'start': 16, 'end': 19, 'multiplier': 1.8},
            'daytime': {'start': 10, 'end': 15, 'multiplier': 1.2},
            'night': {'start': 22, 'end': 6, 'multiplier': 0.4} # Overnight
        }

        demand_multiplier = 1.0
        for period_name, period_info in time_multipliers.items():
            start, end, mult = period_info['start'], period_info['end'], period_info['multiplier']
            if (hour >= start and hour <= end) or \
               (start > end and (hour >= start or hour <= end)): # Handles overnight periods
                demand_multiplier = mult
                break

        # Generate bus positions and passenger counts based on routes
        buses_data = []
        for route in self.routes.itertuples():
            # Estimate number of buses currently on this route based on frequency
            num_buses_on_route = max(1, int(60 / route.frequency))
            for bus_num in range(num_buses_on_route):
                # Simulate bus progress along its route for positioning
                # This is a simplified progress calculation. For true simulation,
                # each bus would have its own state (current stop, next stop, etc.)
                progress = (datetime.now().minute + bus_num * (route.frequency / num_buses_on_route)) / 60
                stop_index_on_route = int(progress * len(route.stops)) % len(route.stops)
                current_stop_id = route.stops[stop_index_on_route]
                current_stop = self.stops[self.stops['stop_id'] == current_stop_id].iloc[0]

                # Calculate passenger count with randomness and time patterns
                base_demand = current_stop['base_demand']
                # Add some randomness to passenger count
                passenger_count = int(base_demand * demand_multiplier * random.uniform(0.8, 1.2))
                passenger_count = min(passenger_count, route.capacity) # Cap at capacity

                buses_data.append({
                    'bus_id': f"R{route.route_id}B{bus_num}",
                    'latitude': current_stop['latitude'],
                    'longitude': current_stop['longitude'],
                    'passenger_count': passenger_count,
                    'route_id': route.route_id,
                    'capacity': route.capacity
                })
        return pd.DataFrame(buses_data)

# ===================== STEP 2: CLUSTER PASSENGER DEMAND ZONES ===================== #

def cluster_demand(data: pd.DataFrame, n_clusters: int = Config.N_CLUSTERS) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Applies K-Means clustering to group high-demand areas, optimizing transport zones.
    It preprocesses the data, finds an optimal number of clusters using silhouette score,
    and visualizes the clusters.

    Args:
        data (pd.DataFrame): DataFrame containing 'latitude', 'longitude', and 'passenger_count'.
        n_clusters (int): The initial number of clusters to try. The function will
                          then iterate to find a better N.

    Returns:
        tuple[pd.DataFrame, np.ndarray]:
            - Clustered data DataFrame with a new 'cluster' column.
            - A NumPy array of cluster centers (latitude, longitude, passenger_count).
    """
    if data.empty:
        print("Warning: No data available for clustering. Returning empty data and None for centers.")
        return data, None
    if len(data) < n_clusters:
        print(f"Warning: Not enough samples ({len(data)}) for {n_clusters} clusters. "
              "Adjusting n_clusters to number of samples.")
        n_clusters = len(data) if len(data) > 1 else 1 # Ensure at least 1 if only 1 sample

    # Features for clustering: geographic coordinates and passenger count
    features = data[['latitude', 'longitude', 'passenger_count']]

    # Preprocess data: Scale features to prevent dominance by larger values (e.g., passenger_count)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Find optimal number of clusters using silhouette score
    # Iterate from 2 clusters up to a maximum of 7, or data size if smaller
    max_clusters_to_try = min(len(data) - 1, 7)
    if max_clusters_to_try < 2:
        print("Not enough data points to perform meaningful clustering for multiple clusters.")
        # If clustering can't be performed, assign all to one cluster
        data['cluster'] = 0
        if not data.empty:
            cluster_centers = np.array([features.mean().values])
        else:
            cluster_centers = np.array([])
        return data, cluster_centers

    best_score = -1
    best_n = n_clusters # Start with the provided n_clusters as default best
    best_kmeans_model = None

    for n in range(2, max_clusters_to_try + 1):
        try:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=Config.KMEANS_N_INIT)
            cluster_labels = kmeans.fit_predict(scaled_features)
            score = silhouette_score(scaled_features, cluster_labels)
            if score > best_score:
                best_score = score
                best_n = n
                best_kmeans_model = kmeans
        except Exception as e:
            print(f"Could not compute silhouette score for {n} clusters: {e}")
            continue

    if best_kmeans_model is None:
        print("Could not find an optimal clustering. Assigning all to one cluster.")
        data['cluster'] = 0
        if not data.empty:
            cluster_centers = np.array([features.mean().values])
        else:
            cluster_centers = np.array([])
        return data, cluster_centers

    # Apply clustering with the optimal number of clusters
    data['cluster'] = best_kmeans_model.predict(scaled_features)
    # Inverse transform cluster centers back to original scale for interpretation
    cluster_centers_original_scale = scaler.inverse_transform(best_kmeans_model.cluster_centers_)

    # Visualize the clusters
    plt.figure(figsize=(12, 8))
    # Scatter plot data points, colored by cluster, size by passenger count
    plt.scatter(data['longitude'], data['latitude'],
                c=data['cluster'], cmap='viridis', alpha=0.7,
                s=data['passenger_count'] / 10 + 5) # Add small constant for visibility of low demand

    # Plot cluster centers as large 'X' marks
    plt.scatter(cluster_centers_original_scale[:, 1], cluster_centers_original_scale[:, 0],
                c='red', marker='X', s=200, label='Cluster Centers', edgecolor='black', linewidth=1.5)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Optimized Public Transport Demand Zones ({best_n} Clusters, Silhouette Score: {best_score:.2f})')
    plt.colorbar(label='Cluster ID')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True) # Ensure output directory exists
    plt.savefig(Config.DEMAND_CLUSTERS_PLOT)
    plt.close()

    return data, cluster_centers_original_scale

# ===================== STEP 3: REINFORCEMENT LEARNING ENVIRONMENT ===================== #

class TransportEnv(gym.Env):
    """
    Custom Reinforcement Learning Environment for optimizing bus routes dynamically.
    The agent learns to adjust bus allocations and redirections based on passenger demand.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, initial_buses_data: pd.DataFrame, cluster_centers: np.ndarray, max_episode_steps: int = Config.MAX_EPISODE_STEPS):
        super(TransportEnv, self).__init__()

        # Store initial data to reset the environment state if needed for new episodes
        self.initial_buses_data = initial_buses_data.copy()
        self.data = initial_buses_data.copy() # Current state of buses

        self.cluster_centers = cluster_centers
        self.num_clusters = len(cluster_centers)

        # Ensure routes are extracted from the initial_buses_data or a separate routes DataFrame
        # For simplicity, assuming initial_buses_data contains valid route_id and capacity
        self.routes_info = self.data[['route_id', 'capacity']].drop_duplicates().set_index('route_id')
        self.routes_ids = self.routes_info.index.tolist()
        self.num_routes = len(self.routes_ids)

        # Define action space: [route_idx, action_type, cluster_idx]
        # route_idx: Index of the route to modify (0 to num_routes - 1)
        # action_type: 0=add bus, 1=remove bus, 2=redirect bus to cluster
        # cluster_idx: Index of the target cluster (0 to num_clusters - 1), only relevant for action_type 2
        self.action_space = spaces.MultiDiscrete([
            self.num_routes,    # Route selection
            3,                   # Action type (add, remove, redirect)
            self.num_clusters   # Cluster selection (for redirect)
        ])

        # Define observation space: [normalized_cluster_passenger_counts, normalized_route_utilization]
        # Each element represents a value from 0 to 1, or potentially higher if not strictly normalized to 1.
        # Here using 0-100 to represent percentages or scaled values.
        self.observation_space = spaces.Box(
            low=0, high=100,
            shape=(self.num_clusters + self.num_routes,),
            dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Initialize current state (will be updated by reset)
        self.state = None
        self.reset() # Call reset to set initial state

    def _get_state(self) -> np.ndarray:
        """
        Calculates the current state of the environment based on:
        1. Normalized passenger counts per cluster.
        2. Normalized utilization of each route.
        """
        # Ensure self.data is not empty before calculations
        if self.data.empty:
            return np.zeros(self.num_clusters + self.num_routes, dtype=np.float32)

        # 1. Cluster passenger counts (normalized)
        cluster_counts = np.zeros(self.num_clusters)

        # Assign each bus to its closest cluster for state calculation
        bus_coords = self.data[['latitude', 'longitude']].values
        if bus_coords.shape[0] > 0 and self.cluster_centers.shape[0] > 0:
            # Calculate distances from each bus to each cluster center
            # Reshape cluster_centers to enable broadcasting for subtraction
            # Sum of squared differences for Euclidean distance
            distances = np.sum((bus_coords[:, np.newaxis, :] - self.cluster_centers[:, :2])**2, axis=2)
            closest_cluster_ids = np.argmin(distances, axis=1)

            for i in range(self.num_clusters):
                buses_in_this_cluster_mask = (closest_cluster_ids == i)
                if np.any(buses_in_this_cluster_mask):
                    cluster_counts[i] = self.data.loc[buses_in_this_cluster_mask, 'passenger_count'].sum()

        # Normalize cluster counts to represent proportion of total demand
        total_cluster_demand = cluster_counts.sum()
        if total_cluster_demand > 0:
            normalized_cluster_counts = (cluster_counts / total_cluster_demand) * 100 # Scale to 0-100
        else:
            normalized_cluster_counts = np.zeros(self.num_clusters)

        # 2. Route utilization (total passengers / total capacity for each route)
        route_utilization = np.zeros(self.num_routes)
        for i, route_id in enumerate(self.routes_ids):
            route_buses = self.data[self.data['route_id'] == route_id]
            if not route_buses.empty:
                current_passengers = route_buses['passenger_count'].sum()
                total_capacity = route_buses['capacity'].sum()
                if total_capacity > 0:
                    utilization = (current_passengers / total_capacity) * 100 # Scale to 0-100
                    route_utilization[i] = utilization
                else:
                    route_utilization[i] = 0.0 # No capacity, 0 utilization

        return np.concatenate([normalized_cluster_counts, route_utilization]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes a chosen action (bus adjustment) and calculates the reward.

        Args:
            action (np.ndarray): An array representing the agent's action
                                 [route_idx, action_type, cluster_idx].

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation (np.ndarray): The new state of the environment.
                - reward (float): The reward received from the action.
                - terminated (bool): Whether the episode ended due to reaching a terminal state.
                - truncated (bool): Whether the episode ended due to truncation (e.g., step limit).
                - info (dict): Additional information for debugging/logging.
        """
        route_idx, action_type, cluster_idx = action
        route_id = self.routes_ids[route_idx] if route_idx < self.num_routes else self.routes_ids[0]
        cluster_center = self.cluster_centers[cluster_idx] if cluster_idx < self.num_clusters else self.cluster_centers[0]

        prev_state = self.state.copy()
        info = {"action_taken": None} # For debugging/logging

        # Apply action
        if action_type == 0:  # Add bus to route
            route_buses = self.data[self.data['route_id'] == route_id]
            if not route_buses.empty:
                # Duplicate an existing bus to simulate adding one
                new_bus = route_buses.sample(1).iloc[0].copy() # Pick a random bus from the route
                new_bus['bus_id'] = f"R{route_id}B{len(route_buses) + random.randint(100,999)}" # Unique ID
                self.data = pd.concat([self.data, pd.DataFrame([new_bus])], ignore_index=True)
                info["action_taken"] = f"Added bus to route {route_id}"
            else:
                info["action_taken"] = f"Attempted to add bus to empty route {route_id}, no action taken."
        elif action_type == 1:  # Remove bus from route
            route_buses = self.data[self.data['route_id'] == route_id]
            if len(route_buses) > 0:
                # Remove a random bus from the route
                bus_to_remove_idx = random.choice(route_buses.index)
                self.data = self.data.drop(bus_to_remove_idx).reset_index(drop=True)
                info["action_taken"] = f"Removed bus from route {route_id}"
            else:
                info["action_taken"] = f"Attempted to remove bus from empty route {route_id}, no action taken."
        elif action_type == 2:  # Redirect bus to cluster
            # This action moves all buses on a given route toward a target cluster center.
            # A more granular approach might be to redirect only a single bus or a subset.
            route_mask = self.data['route_id'] == route_id
            if np.any(route_mask):
                self.data.loc[route_mask, 'latitude'] = cluster_center[0] + random.uniform(-0.005, 0.005) # Add slight noise
                self.data.loc[route_mask, 'longitude'] = cluster_center[1] + random.uniform(-0.005, 0.005)
                info["action_taken"] = f"Redirected buses on route {route_id} to cluster {cluster_idx}"
            else:
                info["action_taken"] = f"Attempted to redirect empty route {route_id}, no action taken."

        # Update state after action
        self.state = self._get_state()

        # Calculate reward based on state changes and optimization goals
        reward = 0.0

        # Reward component 1: Minimize unmet demand (reduce passenger counts in clusters)
        # Assuming lower cluster counts are better (less unmet demand)
        # If prev_state[i] > self.state[i], demand was reduced, which is good.
        demand_reduction_reward = (prev_state[:self.num_clusters] - self.state[:self.num_clusters]).sum()
        reward += demand_reduction_reward * 0.5 # Positive reward for reducing demand

        # Reward component 2: Optimize route utilization (target utilization around 70-90%)
        # Penalize for under- and over-utilization
        target_utilization = 80.0 # Target 80% utilization
        current_utilizations = self.state[self.num_clusters:]
        utilization_penalty = np.sum(np.abs(current_utilizations - target_utilization))
        reward -= utilization_penalty * 0.1 # Penalize deviations from target

        # Reward component 3: Minimize operational cost (penalize excessive buses/actions)
        # This is implicitly handled by the reward for balanced utilization and explicit action penalty
        reward -= 0.05 # Small penalty for each action to encourage efficient decision-making

        self.current_step += 1
        terminated = False # True if episode ends due to specific terminal state (e.g., all demand met)
        truncated = self.current_step >= self.max_episode_steps # True if episode ends due to step limit

        return self.state, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial state. For this simulation, it resets
        the bus data to its original generated state.
        """
        super().reset(seed=seed) # Important for gym.Env
        self.data = self.initial_buses_data.copy()
        self.state = self._get_state()
        self.current_step = 0 # Reset step counter at the start of each episode
        info = {} # Reset info dictionary
        return self.state, info

    def render(self, mode='human'):
        """
        Renders the current state of the environment.
        Currently, this prints the state to the console.
        """
        if mode == 'human':
            print(f"Current State - Clusters (% Demand): {np.round(self.state[:self.num_clusters], 2)}, "
                  f"Route Utilizations (%): {np.round(self.state[self.num_clusters:], 2)}")
        else:
            super().render(mode=mode) # Fallback for other modes

    def visualize_on_map(self, filename: str = Config.INITIAL_ROUTES_MAP) -> folium.Map:
        """
        Creates an interactive Folium map visualization of current bus positions,
        passenger counts, and cluster centers. Saves the map to an HTML file.

        Args:
            filename (str): The name of the HTML file to save the map to.

        Returns:
            folium.Map: The Folium Map object.
        """
        if self.data.empty:
            print(f"No bus data to visualize for {filename}. Returning a basic map.")
            # Return a basic map if no data
            return folium.Map(location=Config.CITY_CENTER, zoom_start=12)

        # Create base map, centered around the mean of current bus locations
        map_center_lat = self.data['latitude'].mean()
        map_center_lon = self.data['longitude'].mean()
        m = folium.Map(location=[map_center_lat, map_center_lon],
                       zoom_start=13, tiles='cartodbpositron')

        # Add cluster centers to the map
        if self.cluster_centers is not None and len(self.cluster_centers) > 0:
            for i, center in enumerate(self.cluster_centers):
                folium.CircleMarker(
                    location=[center[0], center[1]],
                    radius=10,
                    color='purple',
                    fill=True,
                    fill_color='purple',
                    fill_opacity=0.7,
                    popup=f'Cluster {i} Center<br>Passengers: {center[2]:.0f}'
                ).add_to(m)

        # Add bus markers to a MarkerCluster for better performance with many markers
        marker_cluster = MarkerCluster().add_to(m)
        for _, bus in self.data.iterrows():
            # Determine icon color based on utilization (example)
            utilization_ratio = bus['passenger_count'] / bus['capacity'] if bus['capacity'] > 0 else 0
            icon_color = 'green'
            if utilization_ratio > 0.7:
                icon_color = 'orange'
            if utilization_ratio > 0.9:
                icon_color = 'red'

            folium.Marker(
                location=[bus['latitude'], bus['longitude']],
                popup=f"Bus ID: {bus['bus_id']}<br>"
                      f"Route: {bus['route_id']}<br>"
                      f"Passengers: {bus['passenger_count']}/{bus['capacity']} ({utilization_ratio:.1%})",
                icon=folium.Icon(color=icon_color, icon='bus', prefix='fa')
            ).add_to(marker_cluster)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        m.save(filename)
        print(f"Map saved to {filename}")
        return m

# ===================== STEP 4: TRAINING CALLBACK AND RL MODEL ===================== #

class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress in Stable Baselines3.
    Logs mean rewards and plots them over time.
    """
    def __init__(self, check_freq: int = Config.RL_CHECK_FREQ, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        """
        Called at each step during training. Logs mean reward at specified frequency.
        """
        if self.n_calls % self.check_freq == 0:
            # Check if ep_info_buffer is not empty to avoid errors
            if self.model.ep_info_buffer:
                # The ep_info_buffer contains a list of dictionaries, one for each completed episode.
                # 'r' is the episode reward, 'l' is the episode length.
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                self.rewards.append(mean_reward)
                self.timesteps.append(self.n_calls)
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Mean reward = {mean_reward:.2f}")
            else:
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: No complete episode info in buffer yet.")
        return True # Return True to continue training

    def plot_rewards(self, filename: str = Config.TRAINING_PROGRESS_PLOT):
        """
        Plots training rewards over time and saves the plot as a PNG image.
        """
        if not self.rewards:
            print("No rewards data to plot. Training might not have completed any episodes.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.timesteps, self.rewards)
        plt.xlabel('Training Steps')
        plt.ylabel('Mean Reward')
        plt.title('Reinforcement Learning Training Progress')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure output directory exists
        plt.savefig(filename)
        plt.close()
        print(f"Training progress plot saved to {filename}")

def train_rl_model(env: gym.Env, timesteps: int = Config.RL_TIMESTEPS) -> tuple[PPO, TrainingCallback]:
    """
    Trains a PPO (Proximal Policy Optimization) reinforcement learning model
    to optimize bus routes within the given environment.

    Args:
        env (gym.Env): The custom Gym environment (TransportEnv) to train on.
        timesteps (int): The total number of timesteps for training.

    Returns:
        tuple[PPO, TrainingCallback]: The trained PPO model and the training callback instance.
    """
    # Initialize the PPO model with specified hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1, # Verbosity level (0: no output, 1: training progress)
        learning_rate=Config.RL_LEARNING_RATE,
        n_steps=Config.RL_N_STEPS, # Number of steps to run for each environment per update
        batch_size=Config.RL_BATCH_SIZE, # Minibatch size for updating the policy
        n_epochs=Config.RL_N_EPOCHS, # Number of epochs when optimizing the surrogate loss
        gamma=Config.RL_GAMMA, # Discount factor
        gae_lambda=Config.RL_GAE_LAMBDA, # Factor for trade-off between bias and variance for GAE
        clip_range=Config.RL_CLIP_RANGE, # Clipping parameter for PPO
        ent_coef=Config.RL_ENT_COEF, # Entropy coefficient for exploration
        tensorboard_log=Config.OUTPUT_DIR # Log training data to TensorBoard (optional)
    )

    # Initialize the custom callback for progress tracking
    callback = TrainingCallback(check_freq=Config.RL_CHECK_FREQ)

    print(f"Training RL model for {timesteps} timesteps...")
    # Train the model
    model.learn(total_timesteps=timesteps, callback=callback)

    # Save the trained model
    os.makedirs(os.path.dirname(Config.RL_MODEL_SAVE_PATH), exist_ok=True)
    model.save(Config.RL_MODEL_SAVE_PATH)
    print(f"RL model saved to {Config.RL_MODEL_SAVE_PATH}")

    # Plot training progress
    callback.plot_rewards()

    return model, callback

# ===================== STEP 5: MAIN EXECUTION WORKFLOW ===================== #

def setup_simulation_data() -> tuple[pd.DataFrame, np.ndarray, TransportDataGenerator]:
    """
    Sets up the initial simulated public transport data and demand clusters.
    """
    print("\n--- Step 1: Generating simulated transport data ---")
    data_gen = TransportDataGenerator(
        num_buses=Config.NUM_BUSES_INITIAL,
        num_stops=Config.NUM_STOPS_INITIAL
    )
    # Generate data for a specific hour (e.g., morning rush hour)
    initial_buses_data = data_gen.generate_real_time_data(hour=Config.SIMULATION_HOUR)
    print(f"Generated data for {initial_buses_data.shape[0]} buses during {Config.SIMULATION_HOUR} AM.")

    if initial_buses_data.empty:
        raise ValueError("Initial bus data is empty. Cannot proceed with clustering or RL environment setup.")

    print("\n--- Step 2: Analyzing passenger demand clusters ---")
    clustered_data, cluster_centers = cluster_demand(initial_buses_data, n_clusters=Config.N_CLUSTERS)
    if cluster_centers is None or cluster_centers.size == 0:
        raise ValueError("Clustering failed or returned no cluster centers. Cannot proceed.")
    print(f"Identified {len(cluster_centers)} demand clusters.")

    return initial_buses_data, cluster_centers, data_gen

def run_optimization_workflow(initial_buses_data: pd.DataFrame, cluster_centers: np.ndarray) -> tuple[PPO, TransportEnv]:
    """
    Initializes the RL environment and trains the optimization model.
    """
    print("\n--- Step 3: Initializing reinforcement learning environment ---")
    # Pass max_episode_steps to the environment constructor
    env = TransportEnv(initial_buses_data, cluster_centers, max_episode_steps=Config.MAX_EPISODE_STEPS)
    print("Environment initialized.")

    print("\n--- Step 4: Training reinforcement learning model for route optimization ---")
    model, _ = train_rl_model(env, timesteps=Config.RL_TIMESTEPS)
    print("RL model training complete.")

    return model, env

def evaluate_and_visualize_optimized_routes(model: PPO, env: TransportEnv):
    """
    Evaluates the trained model's policy and visualizes the optimized routes.
    """
    print("\n--- Step 5: Evaluating optimized routes and generating visualizations ---")

    # Visualize initial state (before optimization actions are taken by the trained model)
    print("Creating visualization of initial transport network...")
    # Ensure env.data is reset to the true initial state for this visualization
    env.reset()
    env.visualize_on_map(Config.INITIAL_ROUTES_MAP)

    # Apply optimized policy: Run a few steps in the environment using the trained model
    print("Applying trained policy to simulate optimized network...")
    state, info = env.reset() # Reset env to initial state for evaluation
    evaluation_steps = 10 # Run for a few steps to see policy effect
    for i in range(evaluation_steps):
        action, _states = model.predict(state, deterministic=True) # Use deterministic action for evaluation
        state, reward, terminated, truncated, info = env.step(action)
        env.render() # Print current state
        if terminated or truncated:
            print(f"Episode terminated or truncated after {i+1} steps during evaluation.")
            break
    print(f"Simulated {i+1} steps with the optimized policy.")

    # Visualize optimized routes (after applying some actions)
    print("Creating visualization of optimized transport network...")
    env.visualize_on_map(Config.OPTIMIZED_ROUTES_MAP)

    print("\nOptimization complete!")
    print("Results saved to:")
    print(f"- {Config.DEMAND_CLUSTERS_PLOT}: Passenger demand clusters")
    print(f"- {Config.TRAINING_PROGRESS_PLOT}: RL training progress")
    print(f"- {Config.INITIAL_ROUTES_MAP}: Initial bus routes map")
    print(f"- {Config.OPTIMIZED_ROUTES_MAP}: Optimized bus routes map")

def main():
    """
    Main function to orchestrate the public transport optimization workflow.
    """
    print("Starting Public Transport Optimization System...")

    # Ensure output directory exists before any file operations
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    try:
        # 1 & 2: Setup simulation data and demand clusters
        initial_buses_data, cluster_centers, _ = setup_simulation_data()

        # 3 & 4: Initialize RL environment and train the model
        model, env = run_optimization_workflow(initial_buses_data, cluster_centers)

        # 5: Evaluate and visualize the optimized routes
        evaluate_and_visualize_optimized_routes(model, env)

    except ValueError as e:
        print(f"Error during setup: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
