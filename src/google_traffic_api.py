"""
Google Maps API Integration for Real-Time Traffic Data
Replaces simulated traffic generation with real-world data
"""

import pickle
import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import googlemaps
from functools import lru_cache
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

class GoogleTrafficAPI:

    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = 'data/cache'):

        # Get API key - priority: parameter > module variable > environment variable
        if api_key is None:
            api_key = GOOGLE_MAPS_API_KEY if GOOGLE_MAPS_API_KEY else os.getenv('GOOGLE_MAPS_API_KEY')
        
        if api_key is None or api_key == "":
            raise ValueError(
                "Google Maps API key not provided. "
                "Set GOOGLE_MAPS_API_KEY in the project .env file, environment, or api_key parameter. "
                "Get your API key from: https://console.cloud.google.com/google/maps-apis"
            )
        
        self.client = googlemaps.Client(key=api_key)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.last_request_time = 0
        self.min_request_interval = 0.02 
        self.request_count = 0
        self.daily_limit = 40000 
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        if self.request_count >= self.daily_limit:
            raise Exception(f"Daily API limit reached: {self.daily_limit} requests")
    
    def _get_cache_key(self, origin: Tuple[float, float], destination: Tuple[float, float], 
                       departure_time: datetime) -> str:
        """Generate cache key for API request"""
        import hashlib
        key_str = f"{origin[0]}_{origin[1]}_{destination[0]}_{destination[1]}_{departure_time.isoformat()}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached API response if available"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def _cache_response(self, cache_key: str, response: Dict):
        """Cache API response"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w') as f:
            json.dump(response, f)
    
    def get_route_with_traffic(self, 
                               origin: Tuple[float, float], 
                               destination: Tuple[float, float],
                               departure_time: Optional[datetime] = None,
                               use_cache: bool = True) -> Dict:
        """
        Get route with real-time traffic data from Google Maps
        
        Parameters:
        -----------
        origin : tuple (lat, lon)
            Starting point
        destination : tuple (lat, lon)
            Destination point
        departure_time : datetime, optional
            When to depart (for traffic prediction). If None, uses current time
        use_cache : bool
            Whether to use cached responses
        
        Returns:
        --------
        dict with keys:
            - duration_seconds: Travel time in seconds (with traffic)
            - duration_in_traffic_seconds: Travel time considering traffic
            - distance_meters: Distance in meters
            - steps: List of route steps with individual traffic data
        """
        if departure_time is None:
            departure_time = datetime.now()
        
        # Check cache
        cache_key = self._get_cache_key(origin, destination, departure_time)
        if use_cache:
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Get directions with traffic
            directions_result = self.client.directions(
                origin=origin,
                destination=destination,
                mode="driving",
                departure_time=departure_time,
                traffic_model="best_guess",  # Options: best_guess, pessimistic, optimistic
                alternatives=False
            )
            
            if not directions_result:
                raise Exception("No route found")
            
            route = directions_result[0]
            leg = route['legs'][0]
            
            # Extract traffic data
            result = {
                'duration_seconds': leg['duration']['value'],
                'duration_in_traffic_seconds': leg.get('duration_in_traffic', {}).get('value', leg['duration']['value']),
                'distance_meters': leg['distance']['value'],
                'steps': []
            }
            
            # Extract step-by-step data
            for step in leg['steps']:
                result['steps'].append({
                    'distance_meters': step['distance']['value'],
                    'duration_seconds': step['duration']['value'],
                    'start_location': step['start_location'],
                    'end_location': step['end_location'],
                    'html_instructions': step.get('html_instructions', '')
                })
            
            # Cache result
            if use_cache:
                self._cache_response(cache_key, result)
            
            return result
            
        except Exception as e:
            print(f"Error getting route from Google Maps API: {e}")
            # Return fallback (free-flow time estimate)
            return {
                'duration_seconds': 0,
                'duration_in_traffic_seconds': 0,
                'distance_meters': 0,
                'steps': [],
                'error': str(e)
            }
    
    def get_traffic_for_edge(self, 
                            node_u: int, 
                            node_v: int, 
                            G: nx.MultiDiGraph,
                            current_hour: int = None,
                            use_cache: bool = True) -> Dict:
        """
        Get traffic data for a specific edge in the network
        
        Parameters:
        -----------
        node_u : int
            Source node ID
        node_v : int
            Target node ID
        G : networkx.MultiDiGraph
            Road network graph
        current_hour : int, optional
            Current hour (0-23) for traffic prediction
        use_cache : bool
            Whether to use cached responses
        
        Returns:
        --------
        dict with traffic data
        """
        try:
            # Get node coordinates
            node_u_data = G.nodes[node_u]
            node_v_data = G.nodes[node_v]
            
            origin = (node_u_data['y'], node_u_data['x'])  # (lat, lon)
            destination = (node_v_data['y'], node_v_data['x'])
            
            # Calculate departure time
            if current_hour is not None:
                now = datetime.now()
                departure_time = now.replace(hour=current_hour, minute=0, second=0, microsecond=0)
            else:
                departure_time = datetime.now()
            
            # Get route with traffic
            route_data = self.get_route_with_traffic(origin, destination, departure_time, use_cache)
            
            # Calculate congestion level
            if route_data['duration_seconds'] > 0:
                free_flow_time = route_data['duration_seconds']  # Base duration
                traffic_time = route_data.get('duration_in_traffic_seconds', free_flow_time)
                congestion_level = min((traffic_time - free_flow_time) / free_flow_time, 1.0) if free_flow_time > 0 else 0
            else:
                congestion_level = 0
            
            # Calculate speed
            distance_km = route_data['distance_meters'] / 1000
            if route_data['duration_in_traffic_seconds'] > 0:
                speed_kmh = (distance_km / route_data['duration_in_traffic_seconds']) * 3600
            else:
                # Fallback: use edge data
                edge_data = G[node_u][node_v][0]
                length_km = edge_data.get('length', 0) / 1000
                speed_limit = edge_data.get('speed_limit', 50)
                speed_kmh = speed_limit * (1 - congestion_level * 0.5)  # Reduce speed based on congestion
            
            return {
                'travel_time_seconds': route_data.get('duration_in_traffic_seconds', route_data.get('duration_seconds', 0)),
                'free_flow_time_seconds': route_data.get('duration_seconds', 0),
                'distance_meters': route_data.get('distance_meters', 0),
                'congestion_level': congestion_level,
                'speed_kmh': speed_kmh,
                'success': 'error' not in route_data
            }
            
        except Exception as e:
            print(f"Error getting traffic for edge ({node_u}, {node_v}): {e}")
            # Return fallback data
            edge_data = G[node_u][node_v][0]
            length_km = edge_data.get('length', 0) / 1000
            speed_limit = edge_data.get('speed_limit', 50)
            travel_time = (length_km / speed_limit) * 3600
            
            return {
                'travel_time_seconds': travel_time,
                'free_flow_time_seconds': travel_time,
                'distance_meters': edge_data.get('length', 0),
                'congestion_level': 0.0,
                'speed_kmh': speed_limit,
                'success': False,
                'error': str(e)
            }


def save_traffic_data(traffic_df: pd.DataFrame, base_path: str = 'data/processed'):
    """
    Save traffic data to CSV and JSON summary
    """
    os.makedirs(base_path, exist_ok=True)
    
    # Save full data as CSV
    csv_path = f"{base_path}/traffic_data_24h.csv"
    traffic_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved full traffic data: {csv_path}")
    
    # Save compressed version (Parquet for efficiency)
    try:
        parquet_path = f"{base_path}/traffic_data_24h.parquet"
        traffic_df.to_parquet(parquet_path, index=False)
        print(f"✓ Saved compressed data: {parquet_path}")
    except:
        print("  (Parquet save skipped - install pyarrow for compression)")
    
    # Save summary statistics
    summary = {
        'total_records': len(traffic_df),
        'hours_covered': traffic_df['hour'].nunique(),
        'edges_covered': traffic_df[['edge_u', 'edge_v']].drop_duplicates().shape[0],
        'avg_congestion': float(traffic_df['congestion_level'].mean()) if 'congestion_level' in traffic_df.columns else 0.0,
        'avg_speed_kmh': float(traffic_df['actual_speed_kmh'].mean()) if 'actual_speed_kmh' in traffic_df.columns else 0.0,
        'api_success_rate': float(traffic_df['api_success'].mean()) if 'api_success' in traffic_df.columns else 0.0
    }
    
    if 'hour' in traffic_df.columns and 'congestion_level' in traffic_df.columns:
        summary['peak_hour'] = int(traffic_df.groupby('hour')['congestion_level'].mean().idxmax())
        summary['off_peak_hour'] = int(traffic_df.groupby('hour')['congestion_level'].mean().idxmin())
    
    import json
    summary_path = f"{base_path}/traffic_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")
    
    return csv_path


def analyze_traffic_patterns(traffic_df: pd.DataFrame):
    """
    Analyze and print traffic statistics
    """
    print("\n" + "="*60)
    print("TRAFFIC PATTERN ANALYSIS")
    print("="*60)
    
    if 'hour' not in traffic_df.columns:
        print("No hour column found in traffic data")
        return
    
    # Average by hour
    if 'congestion_level' in traffic_df.columns and 'actual_speed_kmh' in traffic_df.columns:
        hourly_avg = traffic_df.groupby('hour').agg({
            'congestion_level': 'mean',
            'actual_speed_kmh': 'mean',
            'travel_time_seconds': 'mean'
        }).round(2)
        
        print("\nAverage Congestion by Hour:")
        for hour, row in hourly_avg.iterrows():
            period = "AM" if hour < 12 else "PM"
            display_hour = hour if hour <= 12 else hour - 12
            if display_hour == 0:
                display_hour = 12
            
            print(f"  {display_hour:2d}:00 {period} - "
                  f"Congestion: {row['congestion_level']:.2f}, "
                  f"Avg Speed: {row['actual_speed_kmh']:.1f} km/h")
    
    # By road type
    if 'road_type' in traffic_df.columns and 'actual_speed_kmh' in traffic_df.columns:
        print("\nAverage Speed by Road Type:")
        road_avg = traffic_df.groupby('road_type')['actual_speed_kmh'].mean().sort_values(ascending=False)
        for road_type, speed in road_avg.head(10).items():
            print(f"  {road_type:20s}: {speed:.1f} km/h")
    
    # Rush hour analysis
    if 'hour' in traffic_df.columns and 'congestion_level' in traffic_df.columns:
        morning_rush = traffic_df[traffic_df['hour'].isin([7, 8, 9])]
        evening_rush = traffic_df[traffic_df['hour'].isin([17, 18, 19])]
        off_peak = traffic_df[traffic_df['hour'].isin([14, 15, 22, 23])]
        
        if len(morning_rush) > 0:
            print(f"\nRush Hour Comparison:")
            print(f"  Morning rush (7-9 AM):")
            print(f"    Avg congestion: {morning_rush['congestion_level'].mean():.2f}")
            if 'actual_speed_kmh' in traffic_df.columns:
                print(f"    Avg speed: {morning_rush['actual_speed_kmh'].mean():.1f} km/h")
            print(f"  Evening rush (5-7 PM):")
            print(f"    Avg congestion: {evening_rush['congestion_level'].mean():.2f}")
            if 'actual_speed_kmh' in traffic_df.columns:
                print(f"    Avg speed: {evening_rush['actual_speed_kmh'].mean():.1f} km/h")
            print(f"  Off-peak:")
            print(f"    Avg congestion: {off_peak['congestion_level'].mean():.2f}")
            if 'actual_speed_kmh' in traffic_df.columns:
                print(f"    Avg speed: {off_peak['actual_speed_kmh'].mean():.1f} km/h")
    
    # API success rate
    if 'api_success' in traffic_df.columns:
        success_rate = traffic_df['api_success'].mean() * 100
        print(f"\nAPI Success Rate: {success_rate:.1f}%")
        if success_rate < 90:
            print("  ⚠ Warning: Low API success rate. Check API key and network connectivity.")


def generate_traffic_from_google_api(G: nx.MultiDiGraph, 
                                     api_key: Optional[str] = None,
                                     hours: int = 24,
                                     current_hour_only: bool = False,
                                     sample_edges: Optional[int] = None,
                                     use_cache: bool = True,
                                     verbose: bool = True) -> pd.DataFrame:
    """
    Generate traffic data using Google Maps API
    
    Parameters:
    -----------
    G : networkx.MultiDiGraph
        Road network graph
    api_key : str, optional
        Google Maps API key
    hours : int
        Number of hours to generate data for (default: 24)
        Ignored if current_hour_only=True
    current_hour_only : bool
        If True, only fetch real-time traffic data for current hour (default: False)
        This uses datetime.now() for real-time traffic data
    sample_edges : int, optional
        If provided, only sample this many edges (for cost control)
        If None, processes all edges
    use_cache : bool
        Whether to use cached API responses
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    pd.DataFrame with traffic data compatible with ACO algorithm
    """
    # Determine which hours to process
    if current_hour_only:
        # Use current hour for real-time traffic data
        current_hour = datetime.now().hour
        hours_to_process = [current_hour]
        if verbose:
            print("="*60)
            print("GENERATING REAL-TIME TRAFFIC DATA FROM GOOGLE MAPS API")
            print("="*60)
            print(f"Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            print(f"🕐 Using current hour: {current_hour}:00 (real-time traffic data)")
            if sample_edges:
                print(f"⚠ Sampling {sample_edges} edges to control API costs")
            print()
    else:
        # Process multiple hours
        hours_to_process = list(range(hours))
        if verbose:
            print("="*60)
            print("GENERATING TRAFFIC DATA FROM GOOGLE MAPS API")
            print("="*60)
            print(f"Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            if sample_edges:
                print(f"⚠ Sampling {sample_edges} edges to control API costs")
            print()
    
    # Initialize API client
    api = GoogleTrafficAPI(api_key=api_key)
    
    # Get edges to process
    all_edges = list(G.edges(keys=True, data=True))
    
    if sample_edges and sample_edges < len(all_edges):
        # Sample edges (prioritize important roads)
        important_edges = []
        other_edges = []
        
        for u, v, key, data in all_edges:
            highway_type = data.get('highway', 'residential')
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            
            # Prioritize main roads
            if highway_type in ['motorway', 'trunk', 'primary', 'secondary']:
                important_edges.append((u, v, key, data))
            else:
                other_edges.append((u, v, key, data))
        
        # Sample: take all important + sample of others
        np.random.seed(42)
        n_important = min(len(important_edges), sample_edges // 2)
        n_other = sample_edges - n_important
        
        sampled_important = important_edges[:n_important]
        sampled_other = list(np.random.choice(len(other_edges), min(n_other, len(other_edges)), replace=False))
        sampled_other = [other_edges[i] for i in sampled_other]
        
        edges_to_process = sampled_important + sampled_other
    else:
        edges_to_process = all_edges
    
    if verbose:
        total_calls = len(edges_to_process) * len(hours_to_process)
        print(f"Processing {len(edges_to_process):,} edges across {len(hours_to_process)} hour(s)...")
        print(f"Estimated API calls: {total_calls:,}")
        print(f"Estimated cost: ${total_calls * 0.005:.2f} (at $0.005 per request)")
        print()
    
    traffic_data = []
    total_edges = len(edges_to_process) * len(hours_to_process)
    processed = 0
    
    for hour in hours_to_process:
        if verbose:
            if current_hour_only:
                print(f"Processing real-time traffic data (current hour: {hour:02d}:00)...")
            else:
                print(f"Processing hour {hour:02d}:00...")
        
        hour_start_time = time.time()
        
        for idx, (u, v, key, data) in enumerate(edges_to_process):
            try:
                # Get traffic data for this edge
                # If current_hour_only, pass None to use datetime.now() for real-time data
                traffic_info = api.get_traffic_for_edge(
                    u, v, G, 
                    current_hour=None if current_hour_only else hour,
                    use_cache=use_cache
                )
                
                # Get edge metadata
                highway_type = data.get('highway', 'residential')
                if isinstance(highway_type, list):
                    highway_type = highway_type[0]
                
                traffic_data.append({
                    'edge_u': u,
                    'edge_v': v,
                    'edge_key': key,
                    'hour': hour,
                    'road_type': highway_type,
                    'length_m': data.get('length', traffic_info.get('distance_meters', 0)),
                    'base_speed_kmh': data.get('speed_limit', 50),
                    'congestion_level': traffic_info.get('congestion_level', 0.0),
                    'actual_speed_kmh': traffic_info.get('speed_kmh', data.get('speed_limit', 50)),
                    'travel_time_seconds': traffic_info.get('travel_time_seconds', 0),
                    'free_flow_time_seconds': traffic_info.get('free_flow_time_seconds', traffic_info.get('travel_time_seconds', 0)),
                    'api_success': traffic_info.get('success', False)
                })
                
                processed += 1
                
            except Exception as e:
                # Add fallback data
                edge_data = G[u][v][0]
                length_km = edge_data.get('length', 0) / 1000
                speed_limit = edge_data.get('speed_limit', 50)
                travel_time = (length_km / speed_limit) * 3600
                
                traffic_data.append({
                    'edge_u': u,
                    'edge_v': v,
                    'edge_key': key,
                    'hour': hour,
                    'road_type': highway_type,
                    'length_m': edge_data.get('length', 0),
                    'base_speed_kmh': speed_limit,
                    'congestion_level': 0.0,
                    'actual_speed_kmh': speed_limit,
                    'travel_time_seconds': travel_time,
                    'free_flow_time_seconds': travel_time,
                    'api_success': False
                })
                processed += 1
        

    
    df = pd.DataFrame(traffic_data)
    
    if verbose:
        print("\n" + "="*60)
        print("TRAFFIC DATA GENERATION COMPLETE")
        print("="*60)
        print(f"Total records: {len(df):,}")
        print(f"Hours covered: {hours}")
        print(f"Edges processed: {df[['edge_u', 'edge_v']].drop_duplicates().shape[0]:,}")
        print(f"API success rate: {df['api_success'].mean()*100:.1f}%")
        print(f"Total API calls: {api.request_count:,}")
    
    return df


def main():
    """
    Main execution - generates traffic data using Google Maps API
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate traffic data using Google Maps API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With API key in environment variable
  $env:GOOGLE_MAPS_API_KEY="your_key"
  python src/google_traffic_api.py --sample-edges 1000

  # With API key as argument
  python src/google_traffic_api.py --api-key "your_key" --sample-edges 500

  # Make sure to set GOOGLE_MAPS_API_KEY environment variable
        """
    )
    parser.add_argument('--api-key', type=str, 
                       help='Google Maps API key (overrides GOOGLE_MAPS_API_KEY variable in file)')
    parser.add_argument('--sample-edges', type=int, default=None, 
                       help='Sample this many edges (to control API costs). If not set, processes all edges.')
    parser.add_argument('--hours', type=int, default=1, 
                       help='Number of hours to generate (default: 1 for real-time). Use 24 for full day analysis.')
    parser.add_argument('--current-hour-only', action='store_true', default=True,
                       help='Only fetch real-time traffic for current hour (default: True). Uses datetime.now() for real-time data.')
    parser.add_argument('--all-hours', action='store_true',
                       help='Generate data for all 24 hours (overrides --current-hour-only)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    args = parser.parse_args()
    
    # Get API key - priority: command line argument > module variable > environment variable
    api_key = args.api_key or (GOOGLE_MAPS_API_KEY if GOOGLE_MAPS_API_KEY else os.getenv('GOOGLE_MAPS_API_KEY'))
    
    if api_key is None or api_key == "":
        print("\n" + "="*60)
        print("⚠ GOOGLE MAPS API KEY NOT FOUND")
        print("="*60)
        print("To use real traffic data from Google Maps API:")
        print("1. Get API key from: https://console.cloud.google.com/google/maps-apis")
        print("2. Set GOOGLE_MAPS_API_KEY variable in src/google_traffic_api.py")
        print("3. Or pass as argument: --api-key 'your_key'")
        print("="*60)
        return 1
    
    # Load network
    print("\nLoading road network...")
    try:
        with open('data/raw/sf_network.pkl', 'rb') as f:
            G = pickle.load(f)
        print(f"✓ Loaded network with {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    except FileNotFoundError:
        print("ERROR: Could not find data/raw/sf_network.pkl")
        print("Please run src/download_network.py first")
        return 1
    
    # Determine if we should use current hour only
    use_current_hour_only = args.current_hour_only and not args.all_hours
    
    # Generate traffic using Google API
    try:
        traffic_df = generate_traffic_from_google_api(
            G,
            api_key=api_key,
            hours=args.hours,
            current_hour_only=use_current_hour_only,
            sample_edges=args.sample_edges,
            use_cache=not args.no_cache,
            verbose=True
        )
    except ValueError as e:
        print(f"\nERROR: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save and analyze traffic data
    save_traffic_data(traffic_df)
    analyze_traffic_patterns(traffic_df)
    
    print("\n" + "="*60)
    print("TRAFFIC DATA GENERATION COMPLETE!")
    print("="*60)
    print("Next step: Run comparison_and_visualization.py")


if __name__ == "__main__":
    main()
