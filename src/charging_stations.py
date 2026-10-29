import requests
import pandas as pd
import pickle
import json
import os
import sys
import osmnx as ox
from shapely.geometry import Point
import geopandas as gpd

sys.stdout.reconfigure(encoding='utf-8')

def get_charging_stations_from_api(latitude=37.7749, longitude=-122.4194, radius_km=15, min_stations=50, max_stations=100):
    
    print(f"Fetching charging stations within {radius_km}km of San Francisco ({latitude}, {longitude})...")
    print(f"Target: {min_stations}-{max_stations} stations")
    
    sf_bounds = {
        'min_lat': 37.770,   # South boundary (matches network)
        'max_lat': 37.805,   # North boundary (matches network)
        'min_lon': -122.435, # West boundary (matches network)
        'max_lon': -122.380  # East boundary (matches network)
    }
    
    print(f"  Filtering to network area: lat {sf_bounds['min_lat']:.3f} to {sf_bounds['max_lat']:.3f}")
    print(f"                          lon {sf_bounds['min_lon']:.3f} to {sf_bounds['max_lon']:.3f}")
    
    url = "https://api.openchargemap.io/v3/poi/"
    all_stations = []
    
    # Try multiple center points to get more stations
    # Only use points within the network bounds to avoid water
    center_points = [
        (37.7875, -122.4075),   # Downtown SF (within bounds)
        (37.7900, -122.4100),   # North SF (within bounds)
        (37.7750, -122.4200),   # Center SF (within bounds)
        (37.7800, -122.4000),   # East SF (within bounds)
        (37.7750, -122.4300),   # West SF (within bounds)
    ]
    
    # Filter center points to only those within network bounds
    valid_centers = []
    for lat, lon in center_points:
        if (sf_bounds['min_lat'] <= lat <= sf_bounds['max_lat'] and
            sf_bounds['min_lon'] <= lon <= sf_bounds['max_lon']):
            valid_centers.append((lat, lon))
    
    center_points = valid_centers if valid_centers else [(37.7875, -122.4075)]
    print(f"  Using {len(center_points)} center points within network bounds")
    
    for center_lat, center_lon in center_points:
        # Use smaller radius to avoid water areas
        search_radius = min(radius_km, 10)  # Cap at 10km to stay within land
        
        params = {
            'latitude': center_lat,
            'longitude': center_lon,
            'distance': search_radius,
            'distanceunit': 'KM',
            'maxresults': 500,
            'compact': 'false',
            'verbose': 'false'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            stations_in_bounds = 0
            stations_outside_bounds = 0
            
            for station in data:
                addr_info = station.get('AddressInfo', {})
                station_lat = addr_info.get('Latitude')
                station_lon = addr_info.get('Longitude')
                
                if not (station_lat and station_lon):
                    continue  # Skip if no coordinates
                
                # Check bounds with strict comparison
                if not (sf_bounds['min_lat'] < station_lat < sf_bounds['max_lat'] and
                        sf_bounds['min_lon'] < station_lon < sf_bounds['max_lon']):
                    stations_outside_bounds += 1
                    continue  # Skip if outside bounds
                
                stations_in_bounds += 1
                
                # Additional validation: check city name if available
                city = addr_info.get('Town', '').lower()
                state = addr_info.get('StateOrProvince', '').lower()
                
                # Prefer stations explicitly in San Francisco
                if city and 'san francisco' not in city and 'sf' not in city:
                    continue  # Skip if city is specified and it's not SF
                
                if not any(s.get('ID') == station.get('ID') for s in all_stations):
                    all_stations.append(station)
            
            # If we have enough stations, stop
            if len(all_stations) >= max_stations:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"  Warning: Error fetching from center ({center_lat:.4f}, {center_lon:.4f}): {e}")
            continue
    
    print(f"\n✓ Retrieved {len(all_stations)} unique charging stations in San Francisco")
    
    if len(all_stations) < min_stations:
        print(f"⚠ Warning: Only found {len(all_stations)} stations (target: {min_stations})")
        print("  This might be due to API limitations or limited station data in the area")
    
    try:
        stations = []
        for station in all_stations:
            # Extract relevant information
            addr_info = station.get('AddressInfo', {})
            connections = station.get('Connections', [])
            
            # Count total charging ports
            num_ports = len(connections)
            
            # Get charging levels/types
            charging_levels = set()
            max_power_kw = 0
            
            for conn in connections:
                level = conn.get('Level', {})
                if level:
                    charging_levels.add(level.get('Title', 'Unknown'))
                
                power = conn.get('PowerKW', 0)
                if power and power > max_power_kw:
                    max_power_kw = power
            
            stations.append({
                'station_id': station.get('ID'),
                'name': addr_info.get('Title', 'Unnamed Station'),
                'latitude': addr_info.get('Latitude'),
                'longitude': addr_info.get('Longitude'),
                'address': addr_info.get('AddressLine1', ''),
                'city': addr_info.get('Town', ''),
                'postcode': addr_info.get('Postcode', ''),
                'num_ports': num_ports,
                'charging_levels': ', '.join(charging_levels),
                'max_power_kw': max_power_kw,
                'operator': station.get('OperatorInfo', {}).get('Title', 'Unknown'),
                'access': station.get('UsageType', {}).get('Title', 'Unknown')
            })
        
        df = pd.DataFrame(stations)
        
        # Filter out stations with no coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Limit to max_stations if we have more
        if len(df) > max_stations:
            df = df.head(max_stations)
            print(f"  Limited to {max_stations} stations")
        
        print(f"✓ Valid stations with coordinates: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error processing data: {e}")
        print("Creating sample dataset instead...")
        return create_sample_stations(num_stations=max_stations)

def create_sample_stations(num_stations=80, G=None):
    
    import numpy as np
    
    print(f"Creating {num_stations} sample charging stations in San Francisco...")
    
    np.random.seed(42)  # For reproducibility
    
    # San Francisco boundaries - matching the network download area
    # This ensures stations are within the actual road network bounds
    sf_bounds = {
        'min_lat': 37.770,   # South boundary (matches network)
        'max_lat': 37.805,   # North boundary (matches network)
        'min_lon': -122.435, # West boundary (matches network)
        'max_lon': -122.380  # East boundary (matches network)
    }
    
    if G is not None:
        node_ids = list(G.nodes())
        num_stations = min(num_stations, len(node_ids))
        selected_nodes = np.random.choice(node_ids, size=num_stations, replace=False)
        latitudes = [G.nodes[node]['y'] for node in selected_nodes]
        longitudes = [G.nodes[node]['x'] for node in selected_nodes]
    else:
        latitudes = np.random.uniform(sf_bounds['min_lat'], sf_bounds['max_lat'], num_stations)
        longitudes = np.random.uniform(sf_bounds['min_lon'], sf_bounds['max_lon'], num_stations)
    
    # Generate realistic station data
    station_ids = list(range(1, num_stations + 1))
    names = [f'EV Station {i}' for i in range(1, num_stations + 1)]
    
    # Vary number of ports (2-8 ports per station)
    num_ports = np.random.choice([2, 3, 4, 4, 4, 5, 6, 6, 8], size=num_stations)
    
    # Vary power levels (50kW, 100kW, 150kW)
    max_power_kw = np.random.choice([50, 50, 50, 100, 100, 150, 150], size=num_stations)
    
    sample_data = {
        'station_id': station_ids,
        'name': names,
        'latitude': list(latitudes),
        'longitude': list(longitudes),
        'num_ports': num_ports.tolist(),
        'max_power_kw': max_power_kw.tolist(),
        'address': [''] * num_stations,
        'city': ['San Francisco'] * num_stations,
        'postcode': [''] * num_stations,
        'charging_levels': ['Level 2, DC Fast'] * num_stations,
        'operator': ['Various'] * num_stations,
        'access': ['Public'] * num_stations
    }
    
    df = pd.DataFrame(sample_data)
    print(f"✓ Created {len(df)} sample stations")
    
    return df

def map_stations_to_network(stations_df, G, max_distance_m=200):
    """
    Map each charging station to nearest road network node
    Filters out stations that are too far from the network (likely in water)
    
    Parameters:
    -----------
    stations_df : pd.DataFrame
        Stations to map
    G : networkx.MultiDiGraph
        Road network graph
    max_distance_m : float
        Maximum distance from station to network node (default: 200m)
        Stations further than this are likely in water and will be filtered out
    """
    
    print("\nMapping stations to road network nodes...")
    mapped_stations = []
    filtered_out = 0
    
    for idx, station in stations_df.iterrows():
        try:
            # Find nearest network node
            nearest_node = ox.distance.nearest_nodes(
                G,
                station['longitude'],
                station['latitude']
            )
            
            # Get node coordinates
            node_data = G.nodes[nearest_node]
            
            # Calculate distance from station to node
            from math import radians, sin, cos, sqrt, atan2
            
            lat1, lon1 = radians(station['latitude']), radians(station['longitude'])
            lat2, lon2 = radians(node_data['y']), radians(node_data['x'])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = 6371000 * c  # Earth radius in meters
            
            # Filter out stations too far from network (likely in water)
            if distance > max_distance_m:
                filtered_out += 1
                continue
            
            mapped_stations.append({
                'station_id': station.get('station_id', idx),
                'name': station.get('name', f'Station {idx}'),
                'latitude': station['latitude'],
                'longitude': station['longitude'],
                'node_id': nearest_node,
                'node_lat': node_data['y'],
                'node_lon': node_data['x'],
                'distance_to_node_m': distance,
                'num_ports': station.get('num_ports', 4),
                'max_power_kw': station.get('max_power_kw', 50),
                'address': station.get('address', ''),
                'operator': station.get('operator', 'Unknown'),
                'charging_levels': station.get('charging_levels', 'Level 2')
            })
        except Exception as e:
            # If mapping fails, skip this station
            filtered_out += 1
            continue
    
    mapped_df = pd.DataFrame(mapped_stations)
    
    # Final validation: double-check all stations are within network bounds
    if len(mapped_df) > 0:
        # Network bounds (same as download_network.py)
        network_bounds = {
            'min_lat': 37.770,
            'max_lat': 37.805,
            'min_lon': -122.435,
            'max_lon': -122.380
        }
        
        # Filter out any stations that somehow got through but are outside bounds
        initial_count = len(mapped_df)
        mapped_df = mapped_df[
            (mapped_df['latitude'] > network_bounds['min_lat']) &
            (mapped_df['latitude'] < network_bounds['max_lat']) &
            (mapped_df['longitude'] > network_bounds['min_lon']) &
            (mapped_df['longitude'] < network_bounds['max_lon'])
        ]
        
        if len(mapped_df) < initial_count:
            print(f"  ⚠ Removed {initial_count - len(mapped_df)} stations outside network bounds in final validation")
    
    print(f"✓ Mapped {len(mapped_df)} stations to network nodes")
    if filtered_out > 0:
        print(f"  Filtered out {filtered_out} stations (too far from network or invalid)")
    if len(mapped_df) > 0:
        print(f"  Average distance to nearest node: {mapped_df['distance_to_node_m'].mean():.1f}m")
    
    return mapped_df

def add_occupancy_simulation(stations_df):
    """
    Add simulated occupancy rates
    In real project, this could come from live APIs or historical data
    """
    import numpy as np
    
    # Simulate base occupancy rates (some stations busier than others)
    np.random.seed(42)  # For reproducibility
    
    stations_df['base_occupancy'] = np.random.beta(2, 5, len(stations_df))
    
    # Stations with more ports tend to have higher utilization
    stations_df['base_occupancy'] = stations_df['base_occupancy'] * (
        1 + 0.1 * (stations_df['num_ports'] / stations_df['num_ports'].max())
    )
    
    # Cap at 0.95
    stations_df['base_occupancy'] = stations_df['base_occupancy'].clip(0, 0.95)
    
    print(f"\n✓ Added occupancy simulation")
    print(f"  Average base occupancy: {stations_df['base_occupancy'].mean():.2%}")
    
    return stations_df

def save_stations(stations_df, base_path='data/processed'):
    """
    Save station data
    """
    import os
    os.makedirs(base_path, exist_ok=True)
    
    # Save as CSV
    csv_path = f"{base_path}/charging_stations.csv"
    stations_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    
    # Save as JSON
    json_path = f"{base_path}/charging_stations.json"
    stations_df.to_json(json_path, orient='records', indent=2)
    print(f"✓ Saved JSON: {json_path}")
    
    return csv_path

def visualize_stations(stations_df, G, save_path='results/stations_map.html'):
    """
    Create interactive map with stations
    """
    import folium
    import os
    os.makedirs('results', exist_ok=True)
    
    # Calculate map center
    center_lat = stations_df['latitude'].mean()
    center_lon = stations_df['longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Add stations
    for idx, station in stations_df.iterrows():
        # Color by power level
        if station.get('max_power_kw', 0) >= 100:
            color = 'red'  # Fast charging
            icon = 'flash'
        elif station.get('max_power_kw', 0) >= 50:
            color = 'orange'  # Medium
            icon = 'bolt'
        else:
            color = 'green'  # Slow
            icon = 'plug'
        
        popup_text = f"""
        <b>{station.get('name', 'Unknown')}</b><br>
        Ports: {station.get('num_ports', 'N/A')}<br>
        Power: {station.get('max_power_kw', 'N/A')} kW<br>
        Latitude: {station.get('latitude', 'N/A')}<br>
        Longitude: {station.get('longitude', 'N/A')}<br>
        Base Occupancy: {station.get('base_occupancy', 0):.1%}
        """
        
        folium.Marker(
            location=[station['latitude'], station['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m)
    
    # Save
    m.save(save_path)
    print(f"\n✓ Saved stations map: {save_path}")
    
    return save_path

def main():
    """
    Main execution
    """
    print("="*60)
    print("EV CHARGING STATIONS DOWNLOAD")
    print("="*60)
    
    # Load road network
    print("\nLoading road network...")
    with open('data/raw/sf_network.pkl', 'rb') as f:
        G = pickle.load(f)
    print(f"✓ Loaded network with {G.number_of_nodes():,} nodes")
    
    # Get charging stations - targeting 20-60 stations in San Francisco
    stations_df = get_charging_stations_from_api(
        latitude=37.7749,      # San Francisco center
        longitude=-122.4194,     # San Francisco center
        radius_km=20,           # 20km radius (covers SF area)
        min_stations=20,         # Target minimum 50 stations
        max_stations=60         # Target maximum 100 stations
    )
    
    mapped_stations = None
    if stations_df.empty:
        cached_stations_path = 'data/processed/charging_stations.json'
        if os.path.exists(cached_stations_path):
            cached_stations = pd.read_json(cached_stations_path)
            required_columns = {'latitude', 'longitude', 'node_id', 'num_ports'}
            if required_columns.issubset(cached_stations.columns) and cached_stations['node_id'].isin(G.nodes).all():
                stations_df = cached_stations
                mapped_stations = cached_stations.copy()
                print(f"Using {len(mapped_stations)} saved charging stations because the data API returned no results.")

    if stations_df.empty:
        stations_df = create_sample_stations(num_stations=60, G=G)

    if mapped_stations is None:
        mapped_stations = map_stations_to_network(stations_df, G)

    if mapped_stations.empty:
        cached_stations_path = 'data/processed/charging_stations.json'
        if os.path.exists(cached_stations_path):
            cached_stations = pd.read_json(cached_stations_path)
            required_columns = {'latitude', 'longitude', 'node_id', 'num_ports'}
            if required_columns.issubset(cached_stations.columns) and cached_stations['node_id'].isin(G.nodes).all():
                stations_df = cached_stations
                mapped_stations = cached_stations.copy()
                print(f"Using {len(mapped_stations)} saved charging stations because fetched stations could not be mapped.")

    if mapped_stations.empty:
        print("No saved stations were usable; generating stations at road nodes.")
        stations_df = create_sample_stations(num_stations=60, G=G)
        mapped_stations = map_stations_to_network(stations_df, G)
    
    # Add occupancy simulation
    mapped_stations = add_occupancy_simulation(mapped_stations)
    
    # Save
    save_stations(mapped_stations)
    
    # Visualize
    visualize_stations(mapped_stations, G)
    
    print("\n" + "="*60)
    print("CHARGING STATIONS DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"Total stations: {len(mapped_stations)}")
    print("Next step: Run src/google_traffic_api.py (requires Google Maps API key)")

if __name__ == "__main__":
    main()