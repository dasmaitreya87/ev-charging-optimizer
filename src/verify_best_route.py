"""
Verification module to validate that ACO returns the best route
Compares ACO results with baseline algorithms and validates cost calculations
"""

import pickle
import pandas as pd
import numpy as np
import networkx as nx
import sys
sys.path.append('src')
from aco_algorithm import ACO_EV_Router


def dijkstra_shortest_path(G, start_node, stations_df, traffic_df, current_hour, w_distance=0.3, w_time=0.4, w_occupancy=0.3):
    """
    Find shortest path using Dijkstra's algorithm (distance only)
    Returns: (path, cost) or (None, inf)
    """
    station_nodes = set(stations_df['node_id'].values)
    best_path = None
    best_cost = float('inf')
    
    for station_node in station_nodes:
        try:
            if nx.has_path(G, start_node, station_node):
                path = nx.shortest_path(G, start_node, station_node, weight='length')
                
                # Calculate cost using same method as ACO
                cost = evaluate_path_cost(G, path, stations_df, traffic_df, current_hour, w_distance, w_time, w_occupancy)
                
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
        except:
            continue
    
    return best_path, best_cost


def fastest_path(G, start_node, stations_df, traffic_df, current_hour, w_distance=0.3, w_time=0.4, w_occupancy=0.3):
    """
    Find fastest path considering traffic (time-based shortest path)
    Returns: (path, cost) or (None, inf)
    """
    # Create graph with time weights
    G_time = G.copy()
    
    # Add time weights to edges
    traffic_lookup = {}
    for _, row in traffic_df.iterrows():
        key = (row['edge_u'], row['edge_v'], row['hour'])
        traffic_lookup[key] = row['travel_time_seconds']
    
    for u, v, data in G_time.edges(data=True):
        key = (u, v, current_hour)
        if key in traffic_lookup:
            data['time_weight'] = traffic_lookup[key]
        else:
            # Fallback: use length and estimated speed
            length = data.get('length', 100)
            speed = data.get('speed_limit', 40)  # km/h
            data['time_weight'] = (length / 1000) / speed * 3600  # seconds
    
    station_nodes = set(stations_df['node_id'].values)
    best_path = None
    best_cost = float('inf')
    
    for station_node in station_nodes:
        try:
            if nx.has_path(G_time, start_node, station_node):
                path = nx.shortest_path(G_time, start_node, station_node, weight='time_weight')
                
                # Calculate cost using same method as ACO
                cost = evaluate_path_cost(G, path, stations_df, traffic_df, current_hour, w_distance, w_time, w_occupancy)
                
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
        except:
            continue
    
    return best_path, best_cost


def evaluate_path_cost(G, path, stations_df, traffic_df, current_hour, w_distance, w_time, w_occupancy):
    """
    Evaluate path cost using the same method as ACO
    This ensures consistent cost calculation
    """
    if path is None or len(path) < 2:
        return float('inf')
    
    # Create traffic lookup
    traffic_lookup = {}
    for _, row in traffic_df.iterrows():
        key = (row['edge_u'], row['edge_v'], row['hour'])
        traffic_lookup[key] = row['travel_time_seconds']
    
    total_distance = 0
    total_time = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        
        if not G.has_edge(u, v):
            return float('inf')
        
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            return float('inf')
        
        # Distance
        total_distance += edge_data.get('length', 0)
        
        # Time (with traffic)
        key = (u, v, current_hour)
        if key in traffic_lookup:
            travel_time = traffic_lookup[key]
        else:
            # Fallback
            length = edge_data.get('length', 100)
            speed = edge_data.get('speed_limit', 40)
            travel_time = (length / 1000) / speed * 3600
        
        if travel_time == float('inf'):
            return float('inf')
        total_time += travel_time
    
    # Station waiting time
    final_station = path[-1]
    station = stations_df[stations_df['node_id'] == final_station]
    if station.empty:
        occupancy = 0.5
    else:
        base_occupancy = station.iloc[0]['base_occupancy']
        if current_hour in [8, 9, 17, 18, 19]:
            occupancy = min(base_occupancy * 1.3, 0.95)
        else:
            occupancy = base_occupancy
    
    waiting_time = occupancy * 900  # Up to 15 min in seconds
    total_time += waiting_time
    
    # Combined cost (same formula as ACO)
    cost = (
        w_distance * (total_distance / 1000) +
        w_time * (total_time / 60) +
        w_occupancy * (waiting_time / 60)
    )
    
    return cost


def verify_aco_result(aco_router, best_path, best_cost, top_routes, verbose=True):
    """
    Verify that ACO returned the best route by:
    1. Comparing with baseline algorithms (Dijkstra, Fastest Path)
    2. Re-evaluating all found routes to ensure consistency
    3. Checking if any route in top_routes has lower cost
    4. Validating cost calculation consistency
    """
    
    if verbose:
        print("\n" + "="*70)
        print("VERIFYING ACO RESULT")
        print("="*70)
    
    G = aco_router.G
    stations_df = aco_router.stations_df
    traffic_df = aco_router.traffic_df
    start_node = aco_router.start_node
    current_hour = aco_router.current_hour
    w_distance = aco_router.w_distance
    w_time = aco_router.w_time
    w_occupancy = aco_router.w_occupancy
    
    issues_found = []
    verification_results = {
        'aco_cost': best_cost,
        'aco_path_length': len(best_path) if best_path else 0,
        'baseline_comparison': {},
        'route_consistency': {},
        'top_routes_check': {}
    }
    
    # 1. Re-evaluate best_path to ensure cost is correct
    if best_path is not None:
        recalculated_cost = aco_router.evaluate_path(best_path)
        if abs(recalculated_cost - best_cost) > 0.01:  # Allow small floating point differences
            issues_found.append(f"Cost mismatch: stored={best_cost:.4f}, recalculated={recalculated_cost:.4f}")
            if verbose:
                print(f"⚠ WARNING: Cost mismatch detected!")
                print(f"   Stored cost: {best_cost:.4f}")
                print(f"   Recalculated cost: {recalculated_cost:.4f}")
                print(f"   Difference: {abs(recalculated_cost - best_cost):.4f}")
        else:
            if verbose:
                print(f"✓ Best path cost verified: {best_cost:.4f}")
    
    # 2. Check if any route in top_routes has lower cost than best_path
    if top_routes:
        for i, (route, cost) in enumerate(top_routes):
            if route is None or cost == float('inf'):
                continue
            
            # Re-evaluate this route
            recalc_cost = aco_router.evaluate_path(route)
            
            if recalc_cost < best_cost - 0.01:  # Allow small tolerance
                issues_found.append(f"Route {i+1} in top_routes has lower cost ({recalc_cost:.4f}) than best_path ({best_cost:.4f})")
                if verbose:
                    print(f"⚠ WARNING: Route {i+1} in top_routes has lower cost!")
                    print(f"   Route {i+1} cost: {recalc_cost:.4f}")
                    print(f"   Best path cost: {best_cost:.4f}")
                    print(f"   Difference: {best_cost - recalc_cost:.4f}")
            else:
                if verbose:
                    print(f"✓ Route {i+1} cost verified: {cost:.4f} (>= best)")
    
    # 3. Compare with baseline algorithms
    if verbose:
        print(f"\nComparing with baseline algorithms...")
    
    # Dijkstra shortest path
    dijkstra_path, dijkstra_cost = dijkstra_shortest_path(
        G, start_node, stations_df, traffic_df, current_hour,
        w_distance, w_time, w_occupancy
    )
    verification_results['baseline_comparison']['dijkstra'] = {
        'path': dijkstra_path,
        'cost': dijkstra_cost,
        'path_length': len(dijkstra_path) if dijkstra_path else 0
    }
    
    if dijkstra_path:
        if verbose:
            print(f"  Dijkstra shortest path:")
            print(f"    Cost: {dijkstra_cost:.4f}")
            print(f"    Path length: {len(dijkstra_path)} nodes")
            print(f"    Station: {dijkstra_path[-1]}")
        
        if dijkstra_cost < best_cost - 0.01:
            issues_found.append(f"Dijkstra found better route: {dijkstra_cost:.4f} vs ACO {best_cost:.4f}")
            if verbose:
                print(f"  ⚠ WARNING: Dijkstra found a better route!")
        else:
            if verbose:
                improvement = ((dijkstra_cost - best_cost) / dijkstra_cost * 100) if dijkstra_cost > 0 else 0
                print(f"  ✓ ACO is {'better' if best_cost < dijkstra_cost else 'equal'} (improvement: {improvement:.2f}%)")
    
    # Fastest path
    fastest_path_result, fastest_cost = fastest_path(
        G, start_node, stations_df, traffic_df, current_hour,
        w_distance, w_time, w_occupancy
    )
    verification_results['baseline_comparison']['fastest'] = {
        'path': fastest_path_result,
        'cost': fastest_cost,
        'path_length': len(fastest_path_result) if fastest_path_result else 0
    }
    
    if fastest_path_result:
        if verbose:
            print(f"  Fastest path (traffic-aware):")
            print(f"    Cost: {fastest_cost:.4f}")
            print(f"    Path length: {len(fastest_path_result)} nodes")
            print(f"    Station: {fastest_path_result[-1]}")
        
        if fastest_cost < best_cost - 0.01:
            issues_found.append(f"Fastest path found better route: {fastest_cost:.4f} vs ACO {best_cost:.4f}")
            if verbose:
                print(f"  ⚠ WARNING: Fastest path found a better route!")
        else:
            if verbose:
                improvement = ((fastest_cost - best_cost) / fastest_cost * 100) if fastest_cost > 0 else 0
                print(f"  ✓ ACO is {'better' if best_cost < fastest_cost else 'equal'} (improvement: {improvement:.2f}%)")
    
    # 4. Check all stations to see if there's a better route
    if verbose:
        print(f"\nChecking all stations for potential better routes...")
    
    station_nodes = set(stations_df['node_id'].values)
    all_station_costs = []
    
    for station_node in station_nodes:
        try:
            if nx.has_path(G, start_node, station_node):
                # Try shortest path to this station
                path = nx.shortest_path(G, start_node, station_node, weight='length')
                cost = evaluate_path_cost(G, path, stations_df, traffic_df, current_hour, w_distance, w_time, w_occupancy)
                
                if cost != float('inf'):
                    all_station_costs.append((station_node, path, cost))
        except:
            continue
    
    if all_station_costs:
        # Sort by cost
        all_station_costs.sort(key=lambda x: x[2])
        best_station_cost = all_station_costs[0][2]
        
        if verbose:
            print(f"  Found {len(all_station_costs)} reachable stations")
            print(f"  Best station (by shortest path): Cost = {best_station_cost:.4f}")
        
        if best_station_cost < best_cost - 0.01:
            issues_found.append(f"Shortest path to best station has lower cost: {best_station_cost:.4f} vs ACO {best_cost:.4f}")
            if verbose:
                print(f"  ⚠ WARNING: There's a station with lower cost!")
                print(f"     Best station cost: {best_station_cost:.4f}")
                print(f"     ACO cost: {best_cost:.4f}")
        else:
            if verbose:
                print(f"  ✓ ACO found optimal or better route")
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)
        
        if issues_found:
            print(f"⚠ {len(issues_found)} issue(s) found:")
            for issue in issues_found:
                print(f"  - {issue}")
        else:
            print("✓ No issues found - ACO result appears correct!")
            print(f"  Best cost: {best_cost:.4f}")
            if dijkstra_path:
                improvement = ((dijkstra_cost - best_cost) / dijkstra_cost * 100) if dijkstra_cost > 0 else 0
                print(f"  Improvement over Dijkstra: {improvement:.2f}%")
            if fastest_path_result:
                improvement = ((fastest_cost - best_cost) / fastest_cost * 100) if fastest_cost > 0 else 0
                print(f"  Improvement over Fastest Path: {improvement:.2f}%")
        print("="*70)
    
    verification_results['issues_found'] = issues_found
    verification_results['is_valid'] = len(issues_found) == 0
    
    return verification_results


def main():
    """
    Example usage: Verify ACO results
    """
    print("="*70)
    print("ACO RESULT VERIFICATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    with open('data/raw/sf_network.pkl', 'rb') as f:
        G = pickle.load(f)
    
    stations_df = pd.read_csv('data/processed/charging_stations.csv')
    traffic_df = pd.read_csv('data/processed/traffic_data_24h.csv')
    
    print(f"✓ Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"✓ Stations: {len(stations_df)}")
    
    # Select random start node
    start_node = np.random.choice(list(G.nodes()))
    current_hour = 8
    
    print(f"\n✓ Start node: {start_node}")
    print(f"✓ Current hour: {current_hour}:00")
    
    # Run ACO
    aco = ACO_EV_Router(
        graph=G,
        stations_df=stations_df,
        traffic_df=traffic_df,
        start_node=start_node,
        current_hour=current_hour,
        battery_level=0.2,
        n_ants=20,
        n_iterations=100
    )
    
    best_path, best_cost, top_routes = aco.run(verbose=True)
    
    if best_path:
        # Verify result
        verification_results = verify_aco_result(aco, best_path, best_cost, top_routes, verbose=True)
        
        if verification_results['is_valid']:
            print("\n✓ Verification passed - ACO result is valid!")
        else:
            print("\n⚠ Verification found issues - please review!")
    else:
        print("\n⚠ No path found - cannot verify")


if __name__ == "__main__":
    main()

