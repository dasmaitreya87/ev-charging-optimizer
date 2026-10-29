"""
Particle Swarm Optimization (PSO) for EV Charging Station Routing

Adapts PSO for discrete routing problems by representing particles as paths
and using path-based velocity updates.
"""

import pickle
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
sys.path.append('src')


class PSO_EV_Router:
    """
    Particle Swarm Optimization for EV charging station routing
    """
    
    def __init__(self,
                 graph,
                 stations_df,
                 traffic_df,
                 start_node,
                 current_hour=8,
                 battery_level=1.0,
                 max_range_km=250,
                 n_particles=30,
                 n_iterations=100,
                 w=0.7,  # Inertia weight
                 c1=1.5,  # Cognitive coefficient
                 c2=1.5,  # Social coefficient
                 w_distance=0.3,
                 w_time=0.4,
                 w_occupancy=0.3):
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
            Current battery level (0-1)
        max_range_km : float
            Maximum EV range in kilometers
        n_particles : int
            Number of particles in the swarm
        n_iterations : int
            Number of iterations
        w : float
            Inertia weight
        c1 : float
            Cognitive coefficient (personal best influence)
        c2 : float
            Social coefficient (global best influence)
        w_distance : float
            Weight for distance objective
        w_time : float
            Weight for time objective
        w_occupancy : float
            Weight for occupancy objective
        """
        
        self.G = graph
        self.stations_df = stations_df
        self.traffic_df = traffic_df
        self.start_node = start_node
        self.current_hour = current_hour
        self.battery_level = battery_level
        self.max_range_km = max_range_km
        
        # PSO parameters
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Objective weights
        self.w_distance = w_distance
        self.w_time = w_time
        self.w_occupancy = w_occupancy
        
        # Preprocess traffic data
        self._preprocess_traffic()
        
        # Get station nodes
        self.station_nodes = set(stations_df['node_id'].values)
        
        # Best solution tracking
        self.best_path = None
        self.best_cost = float('inf')
        self.history = {
            'iteration': [],
            'best_cost': [],
            'avg_cost': [],
            'diversity': []
        }
        
        # Battery parameters
        self.low_battery_threshold = 0.2
        self.critical_battery_threshold = 0.1
        
        # Parallel computing
        self.n_workers = min(os.cpu_count() or 4, n_particles, 8)
    
    def _preprocess_traffic(self):
        """Create fast lookup dict for traffic data"""
        self.traffic_lookup = {}
        for _, row in self.traffic_df.iterrows():
            key = (row['edge_u'], row['edge_v'], row['hour'])
            self.traffic_lookup[key] = {
                'travel_time': row['travel_time_seconds'],
                'congestion': row['congestion_level'],
                'speed': row['actual_speed_kmh']
            }
    
    def get_edge_data(self, u, v):
        """Safely get edge data from MultiDiGraph"""
        if not self.G.has_edge(u, v):
            return None
        edges = self.G.get_edge_data(u, v)
        if edges:
            return list(edges.values())[0]
        return None
    
    def get_traffic_time(self, u, v):
        """Get travel time considering current traffic"""
        key = (u, v, self.current_hour)
        if key in self.traffic_lookup:
            return self.traffic_lookup[key]['travel_time']
        else:
            edge_data = self.get_edge_data(u, v)
            if edge_data is None:
                return float('inf')
            length = edge_data.get('length', 100)
            speed = edge_data.get('speed_limit', 40)
            return (length / 1000) / speed * 3600
    
    def get_station_occupancy(self, node_id):
        """Get station occupancy rate"""
        station = self.stations_df[self.stations_df['node_id'] == node_id]
        if station.empty:
            return 0.5
        base_occupancy = station.iloc[0]['base_occupancy']
        if self.current_hour in [8, 9, 17, 18, 19]:
            return min(base_occupancy * 1.3, 0.95)
        else:
            return base_occupancy
    
    def get_battery_consumption(self, distance_km, speed_kmh=None, traffic_congestion=None, current_battery_level=None):
        """Calculate battery consumption"""
        base_consumption = distance_km / self.max_range_km
        battery_level = current_battery_level if current_battery_level is not None else self.battery_level
        
        if battery_level < self.critical_battery_threshold:
            efficiency_factor = 1.3
        elif battery_level < self.low_battery_threshold:
            efficiency_factor = 1.15
        else:
            efficiency_factor = 1.0
        
        if speed_kmh is not None:
            if speed_kmh > 80:
                speed_factor = 1.2
            elif speed_kmh > 60:
                speed_factor = 1.1
            elif speed_kmh < 20:
                speed_factor = 1.15
            else:
                speed_factor = 1.0
        else:
            speed_factor = 1.0
        
        if traffic_congestion is not None:
            if traffic_congestion > 0.7:
                congestion_factor = 1.25
            elif traffic_congestion > 0.4:
                congestion_factor = 1.1
            else:
                congestion_factor = 1.0
        else:
            congestion_factor = 1.0
        
        total_consumption = base_consumption * efficiency_factor * speed_factor * congestion_factor
        return min(total_consumption, 1.0)
    
    def evaluate_path(self, path):
        """Evaluate total cost of a path"""
        if path is None or len(path) < 2:
            return float('inf')
        
        total_distance = 0
        total_time = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            if not self.G.has_edge(u, v):
                return float('inf')
            
            edge_data = self.get_edge_data(u, v)
            if edge_data is None:
                return float('inf')
            
            total_distance += edge_data.get('length', 0)
            travel_time = self.get_traffic_time(u, v)
            if travel_time == float('inf'):
                return float('inf')
            total_time += travel_time
        
        final_station = path[-1]
        occupancy = self.get_station_occupancy(final_station)
        waiting_time = occupancy * 900
        total_time += waiting_time
        
        cost = (
            self.w_distance * (total_distance / 1000) +
            self.w_time * (total_time / 60) +
            self.w_occupancy * (waiting_time / 60)
        )
        
        return cost
    
    def construct_random_path(self, target_stations):
        """Construct a random valid path to a charging station"""
        current_node = self.start_node
        path = [current_node]
        visited = {current_node}
        battery_remaining = self.battery_level
        max_steps = 500
        steps = 0
        
        # Select random target station
        if not target_stations:
            return None, float('inf')
        
        target_station = np.random.choice(list(target_stations))
        
        while current_node != target_station and steps < max_steps:
            neighbors = list(self.G.neighbors(current_node))
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if not unvisited_neighbors:
                if len(path) > 1:
                    path.pop()
                    visited.remove(current_node)
                    current_node = path[-1]
                    continue
                else:
                    return None, float('inf')
            
            # Filter by battery constraint
            feasible_neighbors = []
            for neighbor in unvisited_neighbors:
                edge_data = self.get_edge_data(current_node, neighbor)
                if edge_data is None:
                    continue
                
                distance_km = edge_data.get('length', 0) / 1000
                traffic_key = (current_node, neighbor, self.current_hour)
                speed_kmh = None
                congestion = None
                if traffic_key in self.traffic_lookup:
                    speed_kmh = self.traffic_lookup[traffic_key]['speed']
                    congestion = self.traffic_lookup[traffic_key]['congestion']
                
                battery_needed = self.get_battery_consumption(
                    distance_km,
                    speed_kmh=speed_kmh,
                    traffic_congestion=congestion,
                    current_battery_level=battery_remaining
                )
                
                safety_margin = 0.05 if battery_remaining < self.low_battery_threshold else 0.0
                if battery_remaining >= (battery_needed + safety_margin):
                    feasible_neighbors.append(neighbor)
            
            if not feasible_neighbors:
                if len(path) > 1:
                    path.pop()
                    visited.remove(current_node)
                    current_node = path[-1]
                    continue
                else:
                    return None, float('inf')
            
            # Prefer neighbors closer to target (greedy with randomness)
            if target_station in feasible_neighbors:
                next_node = target_station
            else:
                # Use heuristic: prefer nodes closer to target
                distances_to_target = []
                for neighbor in feasible_neighbors:
                    try:
                        if nx.has_path(self.G, neighbor, target_station):
                            dist = nx.shortest_path_length(
                                self.G, neighbor, target_station,
                                weight='length', cutoff=50000
                            )
                            distances_to_target.append((neighbor, dist))
                        else:
                            distances_to_target.append((neighbor, float('inf')))
                    except:
                        distances_to_target.append((neighbor, float('inf')))
                
                if distances_to_target:
                    # Select with probability inverse to distance
                    distances_to_target.sort(key=lambda x: x[1])
                    # Take top 3 closest and randomly select
                    top_n = min(3, len(distances_to_target))
                    candidates = [d[0] for d in distances_to_target[:top_n]]
                    next_node = np.random.choice(candidates)
                else:
                    next_node = np.random.choice(feasible_neighbors)
            
            # Update battery
            edge_data = self.get_edge_data(current_node, next_node)
            distance_km = edge_data.get('length', 0) / 1000
            traffic_key = (current_node, next_node, self.current_hour)
            speed_kmh = None
            congestion = None
            if traffic_key in self.traffic_lookup:
                speed_kmh = self.traffic_lookup[traffic_key]['speed']
                congestion = self.traffic_lookup[traffic_key]['congestion']
            
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
        
        if current_node != target_station:
            return None, float('inf')
        
        cost = self.evaluate_path(path)
        return path, cost
    
    def path_crossover(self, path1, path2):
        """Combine two paths by finding common nodes and merging"""
        if path1 is None or path2 is None or len(path1) < 2 or len(path2) < 2:
            return None
        
        # Find common nodes
        common_nodes = set(path1) & set(path2)
        if not common_nodes:
            # No common nodes, return shorter path
            return path1 if len(path1) < len(path2) else path2
        
        # Select a random common node
        common_node = np.random.choice(list(common_nodes))
        
        # Get segments
        idx1 = path1.index(common_node)
        idx2 = path2.index(common_node)
        
        # Combine: path1 start -> common -> path2 end
        new_path = path1[:idx1+1] + path2[idx2+1:]
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for node in new_path:
            if node not in seen:
                seen.add(node)
                result.append(node)
        
        # Ensure path is valid
        if len(result) < 2:
            return path1 if self.evaluate_path(path1) < self.evaluate_path(path2) else path2
        
        # Validate path connectivity
        for i in range(len(result) - 1):
            if not self.G.has_edge(result[i], result[i+1]):
                # Path invalid, return better of original paths
                return path1 if self.evaluate_path(path1) < self.evaluate_path(path2) else path2
        
        return result
    
    def path_mutation(self, path, mutation_rate=0.1):
        """Mutate a path by randomly changing some segments"""
        if path is None or len(path) < 3:
            return path
        
        mutated_path = path.copy()
        
        # Randomly select nodes to mutate (except start and end)
        if np.random.random() < mutation_rate:
            # Try to find alternative path segment
            if len(mutated_path) > 2:
                # Select a random middle node
                idx = np.random.randint(1, len(mutated_path) - 1)
                node = mutated_path[idx]
                
                # Try to find alternative route from previous to next node
                prev_node = mutated_path[idx - 1]
                next_node = mutated_path[idx + 1]
                
                try:
                    # Find alternative path
                    if nx.has_path(self.G, prev_node, next_node):
                        alt_path = nx.shortest_path(
                            self.G, prev_node, next_node,
                            weight='length', cutoff=50000
                        )
                        if len(alt_path) > 2:  # Has intermediate nodes
                            # Replace segment
                            mutated_path = (
                                mutated_path[:idx] +
                                alt_path[1:-1] +  # Exclude endpoints
                                mutated_path[idx+1:]
                            )
                except:
                    pass
        
        # Validate and return
        if self.evaluate_path(mutated_path) != float('inf'):
            return mutated_path
        else:
            return path
    
    def update_particle(self, particle_path, personal_best_path, global_best_path):
        """
        Update particle position using PSO velocity update
        For discrete paths, we use crossover and mutation operations
        """
        if particle_path is None:
            return None
        
        # PSO update: new_position = position + velocity
        # For paths: velocity = w*current + c1*(pbest - current) + c2*(gbest - current)
        # We approximate this with crossover and mutation
        
        r1 = np.random.random()
        r2 = np.random.random()
        
        # Inertia component: keep some of current path
        if np.random.random() < self.w:
            new_path = particle_path.copy()
        else:
            new_path = None
        
        # Cognitive component: move toward personal best
        if personal_best_path is not None and r1 < self.c1 / (self.c1 + self.c2):
            if new_path is None:
                new_path = personal_best_path.copy()
            else:
                # Crossover with personal best
                crossed = self.path_crossover(new_path, personal_best_path)
                if crossed is not None:
                    new_path = crossed
        
        # Social component: move toward global best
        if global_best_path is not None and r2 < self.c2 / (self.c1 + self.c2):
            if new_path is None:
                new_path = global_best_path.copy()
            else:
                # Crossover with global best
                crossed = self.path_crossover(new_path, global_best_path)
                if crossed is not None:
                    new_path = crossed
        
        # Mutation for exploration
        if new_path is not None:
            new_path = self.path_mutation(new_path, mutation_rate=0.1)
        
        # Validate path
        if new_path is None or self.evaluate_path(new_path) == float('inf'):
            # Fallback: return original or personal best
            if personal_best_path is not None:
                return personal_best_path
            return particle_path
        
        return new_path
    
    def run(self, verbose=True):
        """Main PSO loop"""
        
        if self.battery_level <= 0:
            if verbose:
                print("\nERROR: Battery is 0% - cannot route")
            return None, float('inf'), []
        
        if self.battery_level >= 1.0:
            if verbose:
                print("\nINFO: Battery is 100% - no routing needed")
            return None, 0.0, []
        
        if verbose:
            print("\n" + "="*60)
            print("STARTING PSO OPTIMIZATION")
            print("="*60)
            print(f"Parameters:")
            print(f"  Particles: {self.n_particles}")
            print(f"  Iterations: {self.n_iterations}")
            print(f"  Inertia (w): {self.w}")
            print(f"  Cognitive (c1): {self.c1}")
            print(f"  Social (c2): {self.c2}")
            print(f"  Start node: {self.start_node}")
            print(f"  Current hour: {self.current_hour}:00")
            print(f"  Battery level: {self.battery_level:.1%}")
            print(f"  Target stations: {len(self.station_nodes)}")
            print("="*60)
        
        # Filter target stations based on battery level
        if self.battery_level < self.critical_battery_threshold:
            # Find nearest stations
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
            station_distances.sort(key=lambda x: x[1])
            target_stations = [s[0] for s in station_distances[:3]]
        elif self.battery_level < self.low_battery_threshold:
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
            station_distances.sort(key=lambda x: x[1])
            target_stations = [s[0] for s in station_distances[:5]]
        else:
            target_stations = list(self.station_nodes)
        
        # Initialize particles
        particles = []
        personal_bests = []
        personal_best_costs = []
        
        if verbose:
            print("Initializing particles...")
        
        # Initialize particles in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self.construct_random_path, target_stations): i
                for i in range(self.n_particles)
            }
            for future in as_completed(futures):
                path, cost = future.result()
                particles.append(path)
                personal_bests.append(path)
                personal_best_costs.append(cost)
                
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = path
        
        if verbose:
            valid_particles = sum(1 for p in particles if p is not None)
            print(f"✓ Initialized {valid_particles}/{self.n_particles} valid particles")
            print(f"  Initial best cost: {self.best_cost:.4f}")
        
        start_time = time.time()
        
        # Main PSO loop
        for iteration in range(self.n_iterations):
            # Update particles
            for i in range(self.n_particles):
                if particles[i] is None:
                    # Reinitialize failed particle
                    particles[i], cost = self.construct_random_path(target_stations)
                    if cost < personal_best_costs[i]:
                        personal_best_costs[i] = cost
                        personal_bests[i] = particles[i]
                    continue
                
                # Update particle
                new_path = self.update_particle(
                    particles[i],
                    personal_bests[i],
                    self.best_path
                )
                
                if new_path is not None:
                    new_cost = self.evaluate_path(new_path)
                    
                    # Update personal best
                    if new_cost < personal_best_costs[i]:
                        personal_best_costs[i] = new_cost
                        personal_bests[i] = new_path
                    
                    # Update global best
                    if new_cost < self.best_cost:
                        self.best_cost = new_cost
                        self.best_path = new_path
                    
                    particles[i] = new_path
            
            # Track statistics
            valid_costs = [c for c in personal_best_costs if c != float('inf')]
            avg_cost = np.mean(valid_costs) if valid_costs else float('inf')
            
            # Calculate diversity (unique paths)
            unique_paths = len(set(tuple(p) if p else () for p in particles))
            diversity = unique_paths / self.n_particles
            
            self.history['iteration'].append(iteration)
            self.history['best_cost'].append(self.best_cost)
            self.history['avg_cost'].append(avg_cost)
            self.history['diversity'].append(diversity)
            
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
        
        # Get top routes (best path + some alternatives)
        top_routes = []
        if self.best_path is not None:
            top_routes.append((self.best_path, self.best_cost))
            
            # Add some personal bests as alternatives
            sorted_personal = sorted(
                [(p, c) for p, c in zip(personal_bests, personal_best_costs) 
                 if p is not None and c != float('inf')],
                key=lambda x: x[1]
            )
            
            for path, cost in sorted_personal[:2]:
                if path is not None and tuple(path) != tuple(self.best_path):
                    top_routes.append((path, cost))
                    if len(top_routes) >= 3:
                        break
        
        return self.best_path, self.best_cost, top_routes
    
    def get_path_details(self, path):
        """Get detailed information about a path"""
        if path is None:
            return None
        
        details = {
            'nodes': path,
            'num_nodes': len(path),
            'total_distance_km': 0,
            'total_time_min': 0,
            'segments': []
        }
        
        battery_remaining = self.battery_level
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            edge_data = self.get_edge_data(u, v)
            if edge_data is None:
                continue
            
            distance = edge_data.get('length', 0)
            travel_time = self.get_traffic_time(u, v)
            
            traffic_key = (u, v, self.current_hour)
            speed_kmh = None
            congestion = None
            if traffic_key in self.traffic_lookup:
                speed_kmh = self.traffic_lookup[traffic_key]['speed']
                congestion = self.traffic_lookup[traffic_key]['congestion']
            
            battery_consumed = self.get_battery_consumption(
                distance / 1000,
                speed_kmh=speed_kmh,
                traffic_congestion=congestion,
                current_battery_level=battery_remaining
            )
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
        
        details['battery_used'] = sum(seg.get('battery_consumed', 0) for seg in details['segments'])
        details['battery_remaining_at_station'] = battery_remaining
        details['battery_remaining_percent'] = battery_remaining * 100
        details['initial_battery_percent'] = self.battery_level * 100
        
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


if __name__ == "__main__":
    pass
