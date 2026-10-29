import pickle
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, OrderedDict
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import hashlib
import os
import sys
sys.path.append('src')

class ACO_EV_Router:
    
    def __init__(self, 
                 graph, 
                 stations_df,
                 traffic_df,
                 start_node,
                 current_hour=8,
                 battery_level=1.0,
                 max_range_km=250,
                 n_ants=20,
                 n_iterations=100,
                 alpha=1.5,
                 beta=2.0,
                 rho=0.6,
                 Q=100,
                 w_distance=0.3,
                 w_time=0.4,
                 w_occupancy=0.3,
                 tau_min=0.1,
                 tau_max=10.0):
        """
        Parameters:
        -----------
        graph : networkx.MultiDiGraph
            Road network
        stations_df : pd.DataFrame
            Charging stations with node mappings
        traffic_df : pd.DataFrame
            Traffic data for all edges and hours
        start_node : int
            Starting node ID
        current_hour : int
            Current hour of day (0-23)
        battery_level : float
            Current battery level (0-1, where 0.2 = 20%)
        max_range_km : float
            Maximum EV range in kilometers (default: 200km)
        n_ants : int
            Number of ants per iteration
        n_iterations : int
            Number of iterations
        alpha : float
            Pheromone importance
        beta : float
            Heuristic importance
        rho : float
            Pheromone evaporation rate
        Q : float
            Pheromone deposit factor
        w_distance : float
            Weight for distance objective
        w_time : float
            Weight for time/traffic objective
        w_occupancy : float
            Weight for station occupancy objective
        tau_min : float
            Minimum pheromone value (prevents premature convergence)
        tau_max : float
            Maximum pheromone value (prevents numerical issues)
        """
        
        self.G = graph
        self.stations_df = stations_df
        self.traffic_df = traffic_df
        self.start_node = start_node
        self.current_hour = current_hour
        # Validate battery level
        if battery_level <= 0:
            raise ValueError("Battery level cannot be 0% or negative. The car cannot move without battery.")
        # Allow 100% battery but it will be handled in run() method
        
        self.battery_level = battery_level
        self.max_range_km = max_range_km
        
        # ACO parameters
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # Objective weights
        self.w_distance = w_distance
        self.w_time = w_time
        self.w_occupancy = w_occupancy
        
        # Preprocess traffic data for fast lookup
        self._preprocess_traffic()
        
        # Initialize pheromones with better initial values
        # Use a greedy initialization: find shortest path to nearest station
        self.pheromones = defaultdict(lambda: 1.0)
        initial_pheromone = 1.0  # Higher initial value for better exploration
        
        # Try to initialize with a greedy solution
        try:
            # Find nearest station
            station_distances = []
            for station_node in self.station_nodes:
                try:
                    if nx.has_path(self.G, self.start_node, station_node):
                        dist = nx.shortest_path_length(
                            self.G, self.start_node, station_node, 
                            weight='length', cutoff=self.max_range_km * 1000 * 1.5
                        )
                        if dist != float('inf'):
                            station_distances.append((station_node, dist))
                except:
                    continue
            
            if station_distances:
                # Get nearest station
                station_distances.sort(key=lambda x: x[1])
                nearest_station = station_distances[0][0]
                
                # Get shortest path to nearest station
                try:
                    greedy_path = nx.shortest_path(
                        self.G, self.start_node, nearest_station, weight='length'
                    )
                    # Initialize pheromones on greedy path with higher values
                    for i in range(len(greedy_path) - 1):
                        self.pheromones[(greedy_path[i], greedy_path[i+1])] = initial_pheromone * 2.0
                except:
                    pass
        except:
            pass
        
        # Initialize all other edges with base value
        for u, v in self.G.edges():
            if (u, v) not in self.pheromones:
                self.pheromones[(u, v)] = initial_pheromone
        
        # Get station nodes
        self.station_nodes = set(self.stations_df['node_id'].values)
        
        # Precompute distances to stations (major performance improvement)
        # Note: verbose flag not available in __init__, so we'll do it silently
        # and print in run() if needed
        # This might take a moment for large networks, but significantly speeds up pathfinding
        try:
            self._precompute_station_distances(verbose=False)
        except Exception as e:
            # If precomputation fails, initialize empty dict and fall back to on-the-fly computation
            self.dist_to_stations = {}
        
        # Best solution tracking
        self.best_path = None
        self.best_cost = float('inf')
        self.all_found_routes = []  # Track all valid routes found (for debugging)
        self.history = {
            'iteration': [],
            'best_cost': [],
            'avg_cost': [],
            'diversity': []
        }
        
        # Route caching for frequent routes
        self.route_cache = OrderedDict()
        self.cache_max_size = 1000  # Maximum cached routes
        
        # Battery modeling parameters (must be set before _update_battery_efficiency)
        self.low_battery_threshold = 0.2  # 20% - below this, efficiency decreases
        self.critical_battery_threshold = 0.1  # 10% - critical level
        self.battery_efficiency_factor = 1.0  # Will be updated based on battery level
        
        # Initialize battery efficiency factor
        self._update_battery_efficiency()
        
        # Parallel computing (using threading for shared memory)
        # Use threading since we need to share graph and pheromone data
        self.n_workers = min(os.cpu_count() or 4, n_ants, 8)  # Limit to 8 threads max
        self.thread_lock = Lock()  # For thread-safe cache updates
    
    def _preprocess_traffic(self):
        """
        Create fast lookup dict for traffic data
        """
        self.traffic_lookup = {}
        
        for _, row in self.traffic_df.iterrows():
            key = (row['edge_u'], row['edge_v'], row['hour'])
            self.traffic_lookup[key] = {
                'travel_time': row['travel_time_seconds'],
                'congestion': row['congestion_level'],
                'speed': row['actual_speed_kmh']
            }
    
    def _update_battery_efficiency(self):
        """Update battery efficiency factor based on current battery level"""
        if self.battery_level < self.critical_battery_threshold:
            self.battery_efficiency_factor = 0.7  # 30% efficiency loss
        elif self.battery_level < self.low_battery_threshold:
            self.battery_efficiency_factor = 0.85  # 15% efficiency loss
        else:
            self.battery_efficiency_factor = 1.0  # Full efficiency
    
    def _get_cache_key(self, start_node, end_node, battery_level, hour):
        """Generate cache key for route"""
        # Round battery level to 0.05 (5%) increments for cache efficiency
        battery_rounded = round(battery_level * 20) / 20
        key_str = f"{start_node}_{end_node}_{battery_rounded:.2f}_{hour}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_route(self, start_node, end_node):
        """Get cached route if available (thread-safe)"""
        cache_key = self._get_cache_key(start_node, end_node, self.battery_level, self.current_hour)
        
        with self.thread_lock:
            if cache_key in self.route_cache:
                # Move to end (most recently used)
                route = self.route_cache.pop(cache_key)
                self.route_cache[cache_key] = route
                return route
        return None
    
    def _cache_route(self, start_node, end_node, path, cost):
        """Cache a route (thread-safe)"""
        cache_key = self._get_cache_key(start_node, end_node, self.battery_level, self.current_hour)
        
        with self.thread_lock:
            # Remove oldest entry if cache is full
            if len(self.route_cache) >= self.cache_max_size:
                self.route_cache.popitem(last=False)  # Remove oldest
            
            # Add new route
            self.route_cache[cache_key] = {
                'path': path,
                'cost': cost,
                'start_node': start_node,
                'end_node': end_node
            }
    
    def _precompute_station_distances(self, verbose=False):
        """
        Precompute shortest distances from all nodes to stations
        Major performance improvement - done once instead of thousands of times
        Uses a more efficient approach with limited cutoff to avoid long computation
        """
        self.dist_to_stations = {}
        
        # Limit cutoff to reasonable distance (50km) to avoid very long computation
        # For larger networks, this is a trade-off between speed and accuracy
        max_cutoff = min(self.max_range_km * 1000 * 1.5, 50000)  # Max 50km cutoff
        
        # For each station, compute distances to all reachable nodes
        station_list = list(self.station_nodes)
        for idx, station in enumerate(station_list):
            try:
                # Use BFS with cutoff to limit computation
                distances = nx.single_source_shortest_path_length(
                    self.G, station, cutoff=max_cutoff, weight='length'
                )
                
                for node, dist in distances.items():
                    if node not in self.dist_to_stations:
                        self.dist_to_stations[node] = {}
                    self.dist_to_stations[node][station] = dist
                
                if verbose and (idx + 1) % 5 == 0:
                    print(f"  Precomputed {idx + 1}/{len(station_list)} stations...")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not compute distances from station {station}: {e}")
                continue
        
        if verbose:
            print(f"✓ Precomputed distances for {len(self.dist_to_stations)} nodes")
    
    def get_edge_data(self, u, v):
        """
        Safely get edge data from MultiDiGraph
        """
        if not self.G.has_edge(u, v):
            return None
        
        edges = self.G.get_edge_data(u, v)
        if edges:
            # Return first edge's data (or could iterate if multiple edges)
            return list(edges.values())[0]
        return None
    
    def get_traffic_time(self, u, v):

        key = (u, v, self.current_hour)
        
        if key in self.traffic_lookup:
            return self.traffic_lookup[key]['travel_time']
        else:
            edge_data = self.get_edge_data(u, v)
            if edge_data is None:
                return float('inf')
            
            length = edge_data.get('length', 100)
            speed = edge_data.get('speed_limit', 40)
            return (length / 1000) / speed * 3600  # seconds
    
    def get_station_occupancy(self, node_id):
        """
        Get station occupancy rate
        """
        station = self.stations_df[self.stations_df['node_id'] == node_id]
        
        if station.empty:
            return 0.5  # Default
        
        base_occupancy = station.iloc[0]['base_occupancy']
        
        # Adjust by time of day (higher during rush hours)
        if self.current_hour in [8, 9, 17, 18, 19]:
            return min(base_occupancy * 1.3, 0.95)
        else:
            return base_occupancy
    
    def calculate_heuristic(self, current_node, next_node, target_stations):
        """
        Enhanced multi-objective heuristic value (OPTIMIZED)
        Uses precomputed distances and improved cost estimation
        
        Lower cost = higher heuristic value
        """
        
        # 1. Distance component (FAST - uses precomputed distances)
        # Fallback to on-the-fly computation if precomputation didn't cover this node
        if hasattr(self, 'dist_to_stations') and next_node in self.dist_to_stations:
            distances = [
                self.dist_to_stations[next_node].get(s, float('inf')) 
                for s in target_stations
            ]
            if distances:
                distance_to_goal = min(distances) / 1000  # Convert to km
            else:
                distance_to_goal = 10  # Default high value
        else:
            # Fallback: quick estimate using straight-line or small BFS
            # This is slower but ensures we always have a value
            try:
                # Try to get distance to nearest station (with small cutoff for speed)
                min_dist = float('inf')
                for station in target_stations[:5]:  # Limit to first 5 stations for speed
                    try:
                        if nx.has_path(self.G, next_node, station):
                            dist = nx.shortest_path_length(
                                self.G, next_node, station, 
                                weight='length', cutoff=50000  # 50km max
                            )
                            min_dist = min(min_dist, dist)
                    except:
                        continue
                distance_to_goal = min_dist / 1000 if min_dist != float('inf') else 10
            except:
                distance_to_goal = 10  # Default high value
        
        # 2. Traffic/time component (IMPROVED)
        travel_time = self.get_traffic_time(current_node, next_node)
        if travel_time == float('inf'):
            return 0.001  # Very low heuristic for invalid edges
        time_minutes = travel_time / 60
        
        # Get edge data for better estimation
        edge_data = self.get_edge_data(current_node, next_node)
        edge_length_km = (edge_data.get('length', 0) / 1000) if edge_data else 0
        
        # 3. Station occupancy component (if next_node is a station)
        occupancy_cost = 0
        if next_node in target_stations:
            occupancy = self.get_station_occupancy(next_node)
            # Higher occupancy = longer wait time
            occupancy_cost = occupancy * 15  # Up to 15 min wait
        
        # 4. Battery efficiency component (NEW)
        # Prefer edges that don't waste battery
        battery_efficiency = 1.0
        if edge_data:
            # Get traffic data for battery estimation
            traffic_key = (current_node, next_node, self.current_hour)
            if traffic_key in self.traffic_lookup:
                speed_kmh = self.traffic_lookup[traffic_key]['speed']
                congestion = self.traffic_lookup[traffic_key]['congestion']
                # Estimate battery consumption for this edge
                battery_for_edge = self.get_battery_consumption(
                    edge_length_km, speed_kmh=speed_kmh, 
                    traffic_congestion=congestion,
                    current_battery_level=self.battery_level
                )
                # Lower battery consumption = better efficiency
                battery_efficiency = 1.0 / (battery_for_edge + 0.01)
        
        # Adjust weights based on battery level - prioritize distance when battery is low
        w_distance_adj = self.w_distance
        w_time_adj = self.w_time
        w_occupancy_adj = self.w_occupancy
        
        if self.is_critical_battery():
            # Critical battery: ALWAYS prioritize nearest station (distance is almost everything)
            w_distance_adj = 0.95  # 95% weight on distance - strongly prioritize nearest
            w_time_adj = 0.04
            w_occupancy_adj = 0.01
        elif self.is_low_battery():
            # Low battery: strongly prioritize distance - go to nearest station
            w_distance_adj = 0.85  # 85% weight on distance - prioritize nearest
            w_time_adj = 0.10
            w_occupancy_adj = 0.05
        
        # Combined cost with adjusted weights (include battery efficiency)
        total_cost = (
            w_distance_adj * distance_to_goal +
            w_time_adj * time_minutes +
            w_occupancy_adj * occupancy_cost -
            0.1 * battery_efficiency  # Subtracted because higher efficiency = lower cost
        )
        
        # Ensure non-negative
        total_cost = max(0.1, total_cost)
        
        # Heuristic is inverse of cost (higher heuristic = better)
        heuristic = 1.0 / (total_cost + 0.1)
        
        return heuristic
    
    def get_battery_consumption(self, distance_km, speed_kmh=None, traffic_congestion=None, current_battery_level=None):
        """
        Enhanced battery consumption model considering:
        - Battery level (low battery = less efficient)
        - Speed (higher speed = more consumption)
        - Traffic congestion (stop-and-go = more consumption)
        - Distance
        
        Parameters:
        -----------
        distance_km : float
            Distance in kilometers
        speed_kmh : float, optional
            Current speed in km/h
        traffic_congestion : float, optional
            Congestion level (0-1)
        current_battery_level : float, optional
            Current battery level (0-1). If None, uses initial battery_level
        
        Returns battery consumption as fraction (0-1)
        """
        # Base consumption (linear with distance)
        base_consumption = distance_km / self.max_range_km
        
        # Use current battery level if provided, otherwise use initial
        battery_level = current_battery_level if current_battery_level is not None else self.battery_level
        
        # Efficiency factor based on battery level
        # Low battery (< 20%) reduces efficiency due to voltage drop
        if battery_level < self.critical_battery_threshold:
            # Critical battery: 30% efficiency loss
            efficiency_factor = 1.3
        elif battery_level < self.low_battery_threshold:
            # Low battery: 15% efficiency loss
            efficiency_factor = 1.15
        else:
            # Normal battery: full efficiency
            efficiency_factor = 1.0
        
        # Speed factor (if provided)
        # Higher speeds (>60 km/h) consume more energy
        if speed_kmh is not None:
            if speed_kmh > 80:
                speed_factor = 1.2  # 20% more consumption at high speeds
            elif speed_kmh > 60:
                speed_factor = 1.1  # 10% more consumption
            elif speed_kmh < 20:
                speed_factor = 1.15  # 15% more consumption in city traffic
            else:
                speed_factor = 1.0  # Optimal speed range
        else:
            speed_factor = 1.0
        
        # Traffic congestion factor (if provided)
        # Stop-and-go traffic increases consumption
        if traffic_congestion is not None:
            if traffic_congestion > 0.7:  # Heavy congestion
                congestion_factor = 1.25  # 25% more consumption
            elif traffic_congestion > 0.4:  # Moderate congestion
                congestion_factor = 1.1  # 10% more consumption
            else:
                congestion_factor = 1.0
        else:
            congestion_factor = 1.0
        
        # Total consumption with all factors
        total_consumption = base_consumption * efficiency_factor * speed_factor * congestion_factor
        
        # Ensure consumption doesn't exceed 1.0 (100% battery)
        return min(total_consumption, 1.0)
    
    def is_low_battery(self):
        """Check if battery is low"""
        return self.battery_level < self.low_battery_threshold
    
    def is_critical_battery(self):
        """Check if battery is critical"""
        return self.battery_level < self.critical_battery_threshold
    
    def select_next_node(self, current_node, visited, feasible_neighbors, target_stations, ant_id):
        """
        Probabilistically select next node from feasible neighbors
        (feasible = within battery range)
        """
        
        if not feasible_neighbors:
            return None
        
        # When battery is low, make selection more deterministic (favor heuristic more)
        if self.is_low_battery() or self.is_critical_battery():
            # Increase beta (heuristic importance) significantly for low battery
            # This makes the algorithm more deterministic and favor nearest stations
            beta_adj = self.beta * 3.0  # Triple the heuristic importance
            alpha_adj = self.alpha * 0.5  # Reduce pheromone importance
        else:
            beta_adj = self.beta
            alpha_adj = self.alpha
        
        # Calculate probabilities
        probabilities = []
        
        for next_node in feasible_neighbors:
            # Pheromone level
            tau = self.pheromones.get((current_node, next_node), 1.0)
            
            # Heuristic value
            eta = self.calculate_heuristic(current_node, next_node, target_stations)
            
            # Probability with adjusted parameters for low battery
            prob = (tau ** alpha_adj) * (eta ** beta_adj)
            probabilities.append(prob)
        
        # Normalize
        total = sum(probabilities)
        if total == 0:
            probabilities = [1.0 / len(feasible_neighbors)] * len(feasible_neighbors)
        else:
            probabilities = [p / total for p in probabilities]
        
        # When battery is critical, use greedy selection (always pick best heuristic)
        if self.is_critical_battery():
            # Find node with highest probability (best heuristic)
            best_idx = np.argmax(probabilities)
            return feasible_neighbors[best_idx]
        
        # Select next node probabilistically
        next_node = np.random.choice(feasible_neighbors, p=probabilities)
        
        return next_node
    
    def construct_solution(self, ant_id, target_stations):
        """
        Single ant constructs a complete solution
        HYBRID APPROACH: Uses shortest path as baseline, then improves with ACO
        """
        
        # Check cache first for direct routes to stations
        for station_node in target_stations[:5]:  # Check first 5 stations
            cached = self._get_cached_route(self.start_node, station_node)
            if cached and cached['cost'] != float('inf'):
                # Use cached route if battery allows
                cached_path = cached['path']
                if cached_path and len(cached_path) > 1:
                    # Quick check if we have enough battery for cached route
                    total_dist = sum(
                        self.get_edge_data(cached_path[i], cached_path[i+1]).get('length', 0) / 1000
                        for i in range(len(cached_path) - 1)
                        if self.get_edge_data(cached_path[i], cached_path[i+1]) is not None
                    )
                    battery_needed = self.get_battery_consumption(total_dist)
                    if self.battery_level >= battery_needed:
                        return cached_path.copy(), cached['cost']
        
        # HYBRID APPROACH: Try shortest path first (for some ants), then improve
        # Use shortest path for 30% of ants to ensure good baseline
        use_shortest_path = (ant_id % 3 == 0)  # Every 3rd ant uses shortest path
        
        if use_shortest_path:
            # Try shortest path to nearest station
            best_shortest_path = None
            best_shortest_cost = float('inf')
            
            for station_node in target_stations:
                try:
                    if nx.has_path(self.G, self.start_node, station_node):
                        # Try shortest path by distance
                        shortest_path = nx.shortest_path(
                            self.G, self.start_node, station_node, weight='length'
                        )
                        
                        # Check if path is feasible (battery-wise)
                        total_dist = sum(
                            self.get_edge_data(shortest_path[i], shortest_path[i+1]).get('length', 0) / 1000
                            for i in range(len(shortest_path) - 1)
                            if self.get_edge_data(shortest_path[i], shortest_path[i+1]) is not None
                        )
                        battery_needed = self.get_battery_consumption(total_dist)
                        
                        if self.battery_level >= battery_needed:
                            cost = self.evaluate_path(shortest_path)
                            if cost < best_shortest_cost:
                                best_shortest_cost = cost
                                best_shortest_path = shortest_path
                except:
                    continue
            
            if best_shortest_path is not None:
                return best_shortest_path, best_shortest_cost
        
        # For other ants, use ACO probabilistic construction
        current_node = self.start_node
        path = [current_node]
        visited = {current_node}
        battery_remaining = self.battery_level  # Start with current battery level
        
        max_steps = 1000  # Increased for better exploration
        steps = 0
        backtrack_count = 0
        max_backtracks = 10  # Limit backtracking
        
        # Navigate until reaching a charging station
        while current_node not in target_stations and steps < max_steps:
            # Get all neighbors
            neighbors = list(self.G.neighbors(current_node))
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if not unvisited_neighbors:
                # Dead end - try backtracking
                if len(path) > 1 and backtrack_count < max_backtracks:
                    path.pop()
                    visited.remove(current_node)
                    current_node = path[-1]
                    backtrack_count += 1
                    # Restore some battery (rough estimate)
                    battery_remaining = min(self.battery_level, battery_remaining + 0.01)
                    continue
                else:
                    # Too many backtracks or can't backtrack - try shortest path fallback
                    if target_stations:
                        try:
                            # Find nearest reachable station
                            for station_node in target_stations:
                                if nx.has_path(self.G, current_node, station_node):
                                    remaining_path = nx.shortest_path(
                                        self.G, current_node, station_node, weight='length'
                                    )
                                    # Check if we can reach it
                                    remaining_dist = sum(
                                        self.get_edge_data(remaining_path[i], remaining_path[i+1]).get('length', 0) / 1000
                                        for i in range(len(remaining_path) - 1)
                                        if self.get_edge_data(remaining_path[i], remaining_path[i+1]) is not None
                                    )
                                    battery_needed = self.get_battery_consumption(remaining_dist)
                                    if battery_remaining >= battery_needed:
                                        # Combine paths
                                        full_path = path + remaining_path[1:]  # Skip duplicate current_node
                                        cost = self.evaluate_path(full_path)
                                        return full_path, cost
                        except:
                            pass
                    return None, float('inf')
            
            # Filter neighbors by battery constraint
            feasible_neighbors = []
            for neighbor in unvisited_neighbors:
                edge_data = self.get_edge_data(current_node, neighbor)
                if edge_data is None:
                    continue
                
                distance_km = edge_data.get('length', 0) / 1000
                
                # Get traffic data for better battery estimation
                traffic_key = (current_node, neighbor, self.current_hour)
                speed_kmh = None
                congestion = None
                if traffic_key in self.traffic_lookup:
                    speed_kmh = self.traffic_lookup[traffic_key]['speed']
                    congestion = self.traffic_lookup[traffic_key]['congestion']
                
                # Calculate battery consumption with enhanced model
                # Use current battery_remaining for accurate efficiency calculation
                battery_needed = self.get_battery_consumption(
                    distance_km, 
                    speed_kmh=speed_kmh, 
                    traffic_congestion=congestion,
                    current_battery_level=battery_remaining
                )
                
                # Check if we have enough battery (with safety margin for low battery)
                safety_margin = 0.05 if battery_remaining < self.low_battery_threshold else 0.0
                if battery_remaining >= (battery_needed + safety_margin):
                    feasible_neighbors.append(neighbor)
            
            if not feasible_neighbors:
                # No feasible path - battery too low
                # Try backtracking
                if len(path) > 1 and backtrack_count < max_backtracks:
                    path.pop()
                    visited.remove(current_node)
                    current_node = path[-1]
                    backtrack_count += 1
                    # Restore some battery
                    battery_remaining = min(self.battery_level, battery_remaining + 0.01)
                    continue
                else:
                    # Try shortest path fallback from current position
                    if target_stations:
                        try:
                            for station_node in target_stations:
                                if nx.has_path(self.G, current_node, station_node):
                                    remaining_path = nx.shortest_path(
                                        self.G, current_node, station_node, weight='length'
                                    )
                                    remaining_dist = sum(
                                        self.get_edge_data(remaining_path[i], remaining_path[i+1]).get('length', 0) / 1000
                                        for i in range(len(remaining_path) - 1)
                                        if self.get_edge_data(remaining_path[i], remaining_path[i+1]) is not None
                                    )
                                    battery_needed = self.get_battery_consumption(remaining_dist)
                                    if battery_remaining >= battery_needed:
                                        full_path = path + remaining_path[1:]
                                        cost = self.evaluate_path(full_path)
                                        return full_path, cost
                        except:
                            pass
                    return None, float('inf')
            
            # Select next node from feasible neighbors
            next_node = self.select_next_node(
                current_node, visited, feasible_neighbors, target_stations, ant_id
            )
            
            if next_node is None:
                return None, float('inf')
            
            # Update battery
            edge_data = self.get_edge_data(current_node, next_node)
            if edge_data is None:
                return None, float('inf')
            
            distance_km = edge_data.get('length', 0) / 1000
            
            # Get traffic data for accurate battery consumption
            traffic_key = (current_node, next_node, self.current_hour)
            speed_kmh = None
            congestion = None
            if traffic_key in self.traffic_lookup:
                speed_kmh = self.traffic_lookup[traffic_key]['speed']
                congestion = self.traffic_lookup[traffic_key]['congestion']
            
            # Calculate and subtract battery consumption
            # Use current battery_remaining for accurate efficiency calculation
            battery_consumed = self.get_battery_consumption(
                distance_km,
                speed_kmh=speed_kmh,
                traffic_congestion=congestion,
                current_battery_level=battery_remaining
            )
            battery_remaining -= battery_consumed
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
            steps += 1
        
        # Check if valid solution (reached a station)
        if current_node not in target_stations:
            # Last resort: try to find shortest path from current position
            if target_stations:
                try:
                    for station_node in target_stations:
                        if nx.has_path(self.G, current_node, station_node):
                            remaining_path = nx.shortest_path(
                                self.G, current_node, station_node, weight='length'
                            )
                            remaining_dist = sum(
                                self.get_edge_data(remaining_path[i], remaining_path[i+1]).get('length', 0) / 1000
                                for i in range(len(remaining_path) - 1)
                                if self.get_edge_data(remaining_path[i], remaining_path[i+1]) is not None
                            )
                            battery_needed = self.get_battery_consumption(remaining_dist)
                            if battery_remaining >= battery_needed:
                                full_path = path + remaining_path[1:]
                                cost = self.evaluate_path(full_path)
                                return full_path, cost
                except:
                    pass
            return None, float('inf')
        
        # Calculate path cost
        cost = self.evaluate_path(path)
        
        # Cache the route if valid
        if cost != float('inf') and path:
            self._cache_route(self.start_node, current_node, path, cost)
        
        return path, cost
    
    def evaluate_path(self, path):
        """
        Evaluate total cost of a path (weighted sum)
        WITH VALIDATION
        """
        
        if path is None or len(path) < 2:
            return float('inf')
        
        total_distance = 0
        total_time = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Validate edge exists
            if not self.G.has_edge(u, v):
                return float('inf')  # Invalid path
            
            edge_data = self.get_edge_data(u, v)
            if edge_data is None:
                return float('inf')
            
            # Distance
            total_distance += edge_data.get('length', 0)
            
            # Time (with traffic)
            travel_time = self.get_traffic_time(u, v)
            if travel_time == float('inf'):
                return float('inf')
            total_time += travel_time
        
        # Station waiting time
        final_station = path[-1]
        occupancy = self.get_station_occupancy(final_station)
        waiting_time = occupancy * 900  # secondds
        total_time += waiting_time
        
        # Combined cost
        cost = (
            self.w_distance * (total_distance / 1000) +
            self.w_time * (total_time / 60) +
            self.w_occupancy * (waiting_time / 60)
        )

        
        return cost
    
    def update_pheromones(self, all_solutions):

        # Evaporation
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.rho)
        
        # Filter valid solutions and sort by cost
        valid_solutions = [(p, c) for p, c in all_solutions 
                          if p is not None and c != float('inf')]
        
        if not valid_solutions:
            return
        
        # Sort by cost (best first)
        valid_solutions.sort(key=lambda x: x[1])
        
        # ELITISM: Only top 50% of solutions deposit pheromone
        # This focuses search on better regions
        elite_count = max(1, int(len(valid_solutions) * 0.5))
        elite_solutions = valid_solutions[:elite_count]
        
        # Deposit pheromones (more for better solutions)
        for rank, (path, cost) in enumerate(elite_solutions):
            rank_factor = 1.0 / (1.0 + rank * 0.1)  # Decay factor
            
            # Amount of pheromone to deposit
            delta_tau = (self.Q / cost) * rank_factor
            
            # Deposit along path
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                self.pheromones[edge] += delta_tau
        
        # Always deposit extra on best solution (elite ant)
        if valid_solutions:
            best_path, best_cost = valid_solutions[0]
            elite_delta = self.Q / best_cost * 2.0  # Double deposit for best
            for i in range(len(best_path) - 1):
                edge = (best_path[i], best_path[i + 1])
                self.pheromones[edge] += elite_delta
        
        # Apply bounds (IMPORTANT!)
        for edge in self.pheromones:
            self.pheromones[edge] = np.clip(
                self.pheromones[edge], self.tau_min, self.tau_max
            )
    
    def run(self, verbose=True):
        """
        Main ACO loop
        """
        
        # Validate battery level before running
        if self.battery_level <= 0:
            if verbose:
                print("\n" + "="*60)
                print("ERROR: BATTERY IS 0%")
                print("="*60)
                print("The car cannot move. Battery must be above 0% to route to a charging station.")
                print("="*60)
            return None, float('inf'), []
        
        if self.battery_level >= 1.0:
            if verbose:
                print("\n" + "="*60)
                print("INFO: BATTERY IS 100%")
                print("="*60)
                print("The car's battery is fully charged. No charging station routing needed.")
                print("="*60)
            return None, 0.0, []  # Return no path with zero cost (no need to charge)
        
        if verbose:
            print("\n" + "="*60)
            print("STARTING ACO OPTIMIZATION")
            print("="*60)
            print(f"Parameters:")
            print(f"  Ants: {self.n_ants}")
            print(f"  Iterations: {self.n_iterations}")
            print(f"  Alpha (pheromone): {self.alpha}")
            print(f"  Beta (heuristic): {self.beta}")
            print(f"  Rho (evaporation): {self.rho}")
            print(f"  Start node: {self.start_node}")
            print(f"  Current hour: {self.current_hour}:00")
            print(f"  Battery level: {self.battery_level:.1%}")
            if self.is_low_battery():
                print(f"  ⚠ LOW BATTERY WARNING (efficiency: {self.battery_efficiency_factor:.1%})")
            if self.is_critical_battery():
                print(f"  🚨 CRITICAL BATTERY WARNING (efficiency: {self.battery_efficiency_factor:.1%})")
            print(f"  Max range: {self.max_range_km} km")
            print(f"  Target stations: {len(self.station_nodes)}")
            print(f"  Parallel workers: {self.n_workers}")
            print(f"  Route cache size: {len(self.route_cache)}")
            print("="*60)
        
        # When battery is low, filter to only nearest stations to ensure we go to closest one
        if self.is_low_battery() or self.is_critical_battery():
            # Find nearest stations to start node
            station_distances = []
            for station_node in self.station_nodes:
                try:
                    if nx.has_path(self.G, self.start_node, station_node):
                        # Use precomputed distance if available
                        if hasattr(self, 'dist_to_stations') and self.start_node in self.dist_to_stations:
                            dist = self.dist_to_stations[self.start_node].get(station_node, float('inf'))
                        else:
                            # Fallback: compute distance
                            dist = nx.shortest_path_length(
                                self.G, self.start_node, station_node, 
                                weight='length', cutoff=self.max_range_km * 1000 * 1.5
                            )
                        
                        if dist != float('inf'):
                            station_distances.append((station_node, dist))
                except:
                    continue
            
            # Sort by distance and keep only nearest stations
            station_distances.sort(key=lambda x: x[1])
            
            if self.is_critical_battery():
                # Critical: only consider nearest 2-3 stations
                max_stations = 3
            else:
                # Low battery: consider nearest 5 stations
                max_stations = 5
            
            if station_distances:
                target_stations = [s[0] for s in station_distances[:max_stations]]
                if verbose:
                    print(f"  ⚠ Low battery mode: Filtered to {len(target_stations)} nearest stations")
                    for i, (station, dist) in enumerate(station_distances[:max_stations], 1):
                        print(f"    {i}. Station {station}: {dist/1000:.2f} km away")
            else:
                # Fallback: use all stations if we can't compute distances
                target_stations = list(self.station_nodes)
        else:
            target_stations = list(self.station_nodes)
        
        start_time = time.time()
        no_improvement_count = 0
        no_improvement_threshold = 60  # Increased threshold for better convergence
        
        # Track top routes per station (to ensure we have routes to same station)
        routes_by_station = {}  # {station_node: [(path, cost), ...]}
        
        for iteration in range(self.n_iterations):
            all_solutions = []
            
            # Parallel ant construction using threading
            if self.n_workers > 1 and self.n_ants > 1:
                # Use ThreadPoolExecutor for parallel ant construction
                # Threading is better here since we share graph and pheromone data
                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    # Submit all ant tasks
                    futures = {
                        executor.submit(self.construct_solution, ant_id, target_stations): ant_id
                        for ant_id in range(self.n_ants)
                    }
                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            path, cost = future.result()
                            all_solutions.append((path, cost))
                        except Exception as e:
                            # If an ant fails, add invalid solution
                            all_solutions.append((None, float('inf')))
            else:
                # Sequential construction (fallback or single worker)
                for ant_id in range(self.n_ants):
                    path, cost = self.construct_solution(ant_id, target_stations)
                    all_solutions.append((path, cost))
            
            # Update best solution and track top routes
            for path, cost in all_solutions:
                # Track all valid routes for debugging
                if path is not None and cost != float('inf') and len(path) > 0:
                    # Re-evaluate cost to ensure consistency
                    recalc_cost = self.evaluate_path(path)
                    if abs(recalc_cost - cost) > 0.01:
                        # Cost mismatch - use recalculated cost
                        cost = recalc_cost
                    
                    # Update best if this is better
                    if cost < self.best_cost:
                        self.best_cost = cost
                        self.best_path = path
                        no_improvement_count = 0  # Reset counter
                    
                    # Track all found routes (for debugging/verification)
                    self.all_found_routes.append((path.copy(), cost, iteration))
                
                # Track top routes per station (only valid paths)
                if path is not None and cost != float('inf') and len(path) > 0:
                    target_station = path[-1]  # Last node is the target station
                    
                    # Initialize station entry if needed
                    if target_station not in routes_by_station:
                        routes_by_station[target_station] = []
                    
                    # Check if this path is already tracked for this station (avoid duplicates)
                    path_tuple = tuple(path)
                    is_duplicate = any(tuple(p) == path_tuple for p, _ in routes_by_station[target_station])
                    
                    if not is_duplicate:
                        routes_by_station[target_station].append((path, cost))
                        # Keep only top 10 routes per station (we'll filter to top 3 at the end)
                        routes_by_station[target_station].sort(key=lambda x: x[1])
                        routes_by_station[target_station] = routes_by_station[target_station][:10]
            
            # Update pheromones
            self.update_pheromones(all_solutions)
            
            # Track statistics
            valid_costs = [c for p, c in all_solutions if c != float('inf')]
            avg_cost = np.mean(valid_costs) if valid_costs else float('inf')
            
            # Calculate diversity (unique paths)
            unique_paths = len(set(tuple(p) if p else () for p, c in all_solutions))
            diversity = unique_paths / self.n_ants
            
            self.history['iteration'].append(iteration)
            self.history['best_cost'].append(self.best_cost)
            self.history['avg_cost'].append(avg_cost)
            self.history['diversity'].append(diversity)
            
            # Check for convergence
            if iteration > 0:
                if self.history['best_cost'][-1] >= self.history['best_cost'][-2]:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
            
            # Early stopping
            if no_improvement_count >= no_improvement_threshold:
                if verbose:
                    print(f"Early stopping at iteration {iteration} (no improvement for {no_improvement_threshold} iterations)")
                break
            
            # Progress output
            if verbose and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                elapsed = time.time() - start_time
                print(f"Iter {iteration:3d} | "
                      f"Best: {self.best_cost:7.2f} | "
                      f"Avg: {avg_cost:7.2f} | "
                      f"Diversity: {diversity:.2f} | "
                      f"Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        
        if verbose:
            print("="*60)
            print(f"OPTIMIZATION COMPLETE in {total_time:.1f}s")
            print(f"Best cost found: {self.best_cost:.2f}")
            print(f"Best path length: {len(self.best_path) if self.best_path else 0} nodes")
            print("="*60)
        
        # Return best path and top 3 routes to the SAME station
        # First, identify which station the best route goes to
        if self.best_path is not None and len(self.best_path) > 0:
            best_station = self.best_path[-1]  # Last node is the target station
            
            # Get all routes to the best station
            if best_station in routes_by_station:
                routes_to_same_station = routes_by_station[best_station]
            else:
                routes_to_same_station = []
            
            # Filter out the best_path itself (we'll add it first)
            routes_to_same_station = [
                (p, c) for p, c in routes_to_same_station 
                if p is not None and tuple(p) != tuple(self.best_path)
            ]
            
            # Sort by cost and keep top 2 (best_path will be added first)
            routes_to_same_station.sort(key=lambda x: x[1])
            routes_to_same_station = routes_to_same_station[:2]  # Keep top 2 additional routes
            
            # Build final list: best_path first, then top 2 alternative routes to same station
            final_top_routes = [(self.best_path, self.best_cost)]
            final_top_routes.extend(routes_to_same_station)
            final_top_routes = final_top_routes[:3]  # Ensure max 3 routes
            
            # If we don't have 3 routes yet, try to find alternative paths
            if len(final_top_routes) < 3:
                if verbose:
                    print(f"  Note: Only found {len(final_top_routes)} route(s) to station {best_station}")
                    print(f"  Attempting to find alternative paths...")
                
                # Try to find alternative paths by finding shortest paths with some edges from best path temporarily removed
                try:
                    # Method: Find shortest path, then find another path avoiding some edges from the first
                    if len(self.best_path) > 2:
                        # Try to find alternative by avoiding middle portion of best path
                        avoid_edges = set()
                        mid_point = len(self.best_path) // 2
                        for i in range(max(1, mid_point - 2), min(mid_point + 2, len(self.best_path) - 1)):
                            avoid_edges.add((self.best_path[i], self.best_path[i+1]))
                        
                        # Create temporary graph without those edges
                        G_temp = self.G.copy()
                        for u, v in list(avoid_edges):
                            if G_temp.has_edge(u, v):
                                G_temp.remove_edge(u, v)
                        
                        # Try to find path in modified graph
                        if nx.has_path(G_temp, self.start_node, best_station):
                            alt_path = nx.shortest_path(G_temp, self.start_node, best_station, weight='length')
                            path_tuple = tuple(alt_path)
                            is_duplicate = any(tuple(p) == path_tuple for p, _ in final_top_routes)
                            
                            if not is_duplicate:
                                alt_cost = self.evaluate_path(alt_path)
                                if alt_cost != float('inf'):
                                    final_top_routes.append((alt_path, alt_cost))
                    
                    # If still need more, try one more alternative
                    if len(final_top_routes) < 3:
                        # Try using time-based shortest path instead of distance
                        try:
                            alt_path2 = nx.shortest_path(self.G, self.start_node, best_station, weight='length')
                            path_tuple2 = tuple(alt_path2)
                            is_duplicate2 = any(tuple(p) == path_tuple2 for p, _ in final_top_routes)
                            
                            if not is_duplicate2:
                                alt_cost2 = self.evaluate_path(alt_path2)
                                if alt_cost2 != float('inf'):
                                    final_top_routes.append((alt_path2, alt_cost2))
                        except:
                            pass
                    
                    # Sort by cost again
                    final_top_routes.sort(key=lambda x: x[1])
                    final_top_routes = final_top_routes[:3]
                    
                    if verbose and len(final_top_routes) >= 3:
                        print(f"  ✓ Found {len(final_top_routes)} routes (including alternatives)")
                except Exception as e:
                    if verbose:
                        print(f"  Could not find alternative paths: {e}")
            
            top_routes = final_top_routes
        else:
            # If no best path, try to find best station from routes_by_station
            if routes_by_station:
                # Find station with best route
                best_station = None
                best_cost = float('inf')
                for station, routes in routes_by_station.items():
                    if routes and routes[0][1] < best_cost:
                        best_cost = routes[0][1]
                        best_station = station
                
                if best_station and best_station in routes_by_station:
                    top_routes = routes_by_station[best_station][:3]
                else:
                    top_routes = []
            else:
                top_routes = []
        
        # Return format: (best_path, best_cost, top_3_routes)
        # top_3_routes is a list of (path, cost) tuples, all going to the same station
        return self.best_path, self.best_cost, top_routes
    
    def get_path_details(self, path):
        """
        Get detailed information about a path
        """
        
        if path is None:
            return None
        
        details = {
            'nodes': path,
            'num_nodes': len(path),
            'total_distance_km': 0,
            'total_time_min': 0,
            'segments': []
        }
        
        # Track battery level throughout the route
        battery_remaining = self.battery_level
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            edge_data = self.get_edge_data(u, v)
            if edge_data is None:
                continue
            
            distance = edge_data.get('length', 0)
            travel_time = self.get_traffic_time(u, v)
            
            # Get traffic data for accurate battery consumption
            traffic_key = (u, v, self.current_hour)
            speed_kmh = None
            congestion = None
            if traffic_key in self.traffic_lookup:
                speed_kmh = self.traffic_lookup[traffic_key]['speed']
                congestion = self.traffic_lookup[traffic_key]['congestion']
            
            # Calculate battery consumption using current battery level for accurate efficiency
            battery_consumed = self.get_battery_consumption(
                distance / 1000,
                speed_kmh=speed_kmh,
                traffic_congestion=congestion,
                current_battery_level=battery_remaining
            )
            
            # Update remaining battery
            battery_remaining = max(0.0, battery_remaining - battery_consumed)
            
            details['total_distance_km'] += distance / 1000
            details['total_time_min'] += travel_time / 60
            
            details['segments'].append({
                'from_node': u,
                'to_node': v,
                'distance_m': distance,
                'time_sec': travel_time,
                'speed_kmh': speed_kmh if speed_kmh else edge_data.get('speed_limit', 50),
                'congestion_level': congestion if congestion is not None else 0.0,
                'battery_consumed': battery_consumed,
                'battery_remaining': battery_remaining,
                'battery_remaining_percent': battery_remaining * 100
            })
        
        # Add battery usage to details
        details['battery_used'] = sum(seg.get('battery_consumed', 0) for seg in details['segments'])
        details['battery_remaining_at_station'] = battery_remaining
        details['battery_remaining_percent'] = battery_remaining * 100
        details['initial_battery_percent'] = self.battery_level * 100
        
        # Add station info
        final_station_node = path[-1]
        station_info = self.stations_df[
            self.stations_df['node_id'] == final_station_node
        ]
        
        if not station_info.empty:
            station = station_info.iloc[0]
            details['station'] = {
                'name': station['name'],
                'address': station.get('address', 'N/A'),
                'num_ports': station['num_ports'],
                'occupancy': self.get_station_occupancy(final_station_node)
            }
        
        return details
    
    def get_all_found_routes(self):
        """
        Get all routes found during optimization (for debugging/verification)
        Returns list of (path, cost, iteration) tuples
        """
        return self.all_found_routes.copy()
    
    def verify_best_route(self, verbose=True):
        """
        Verify that best_path is actually the best route among all found routes
        """
        if self.best_path is None:
            if verbose:
                print("No best path to verify")
            return False
        
        if verbose:
            print(f"\nVerifying best route among {len(self.all_found_routes)} found routes...")
        
        # Check all found routes
        issues = []
        for path, cost, iteration in self.all_found_routes:
            if path is None:
                continue
            
            # Re-evaluate to ensure consistency
            recalc_cost = self.evaluate_path(path)
            
            if recalc_cost < self.best_cost - 0.01:  # Allow small tolerance
                issues.append({
                    'path': path,
                    'cost': recalc_cost,
                    'iteration': iteration,
                    'station': path[-1] if path else None
                })
        
        if issues:
            if verbose:
                print(f"⚠ Found {len(issues)} route(s) with lower cost than best_path:")
                for issue in issues[:5]:  # Show first 5
                    print(f"  - Cost: {issue['cost']:.4f} (vs best: {self.best_cost:.4f}), "
                          f"Station: {issue['station']}, Iteration: {issue['iteration']}")
            return False
        else:
            if verbose:
                print(f"✓ Best route verified - no better routes found")
            return True

def main():
    """
    Example usage
    """
    print("="*60)
    print("ACO EV CHARGING STATION ROUTING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    with open('data/raw/sf_network.pkl', 'rb') as f:
        G = pickle.load(f)
    
    stations_df = pd.read_csv('data/processed/charging_stations.csv')
    traffic_df = pd.read_csv('data/processed/traffic_data_24h.csv')
    
    print(f"✓ Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"✓ Stations: {len(stations_df)}")
    print(f"✓ Traffic records: {len(traffic_df):,}")
    
    # Select random start node
    start_node = np.random.choice(list(G.nodes()))
    print(f"\n✓ Start node: {start_node}")
    
    # Run ACO
    aco = ACO_EV_Router(
        graph=G,
        stations_df=stations_df,
        traffic_df=traffic_df,
        start_node=start_node,
        current_hour=8,
        battery_level=0.2,
        max_range_km=200, 
        n_ants=20,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.5
    )
    
    best_path, best_cost, top_routes = aco.run(verbose=True)
    
    # Show results
    if best_path:
        details = aco.get_path_details(best_path)
        print("\nBest Route Details:")
        print(f"  Total distance: {details['total_distance_km']:.2f} km")
        print(f"  Total time: {details['total_time_min']:.1f} minutes")
        print(f"  Battery used: {details.get('battery_used', 0):.1%}")
        print(f"  Number of segments: {len(details['segments'])}")
        if 'station' in details:
            print(f"  Destination: {details['station']['name']}")
            print(f"  Station occupancy: {details['station']['occupancy']:.1%}")

if __name__ == "__main__":
    main()