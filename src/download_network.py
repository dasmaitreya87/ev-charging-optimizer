import osmnx as ox
import networkx as nx
import pickle
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Configure OSMnx
ox.settings.log_console = True
ox.settings.use_cache = True

def download_sf_network(method='bbox'):
    
    if method == 'bbox':
        north, south = 37.805, 37.770
        east, west = -122.380, -122.435        
        G = ox.graph_from_bbox(
            north, south, east, west, 
            network_type='drive',
            simplify=True
        )
        
    elif method == 'radius':
        center_point = (37.7749, -122.4194)
        radius = 5000  # meters
        
        print(f"Downloading road network within {radius}m of {center_point}")
        
        G = ox.graph_from_point(
            center_point, 
            dist=radius,
            network_type='drive',
            simplify=True
        )
        
    else:  # place
        place_name = "Downtown, San Francisco, California, USA"
        
        print(f"Downloading road network for {place_name}")
        
        G = ox.graph_from_place(
            place_name,
            network_type='drive',
            simplify=True
        )
    
    # Print network statistics
    print(f"\n{'='*50}")
    print(f"Network Statistics:")
    print(f"{'='*50}")
    print(f"Number of nodes: {G.number_of_nodes():,}")
    print(f"Number of edges: {G.number_of_edges():,}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # Calculate total network length
    total_length = sum([data['length'] for u, v, data in G.edges(data=True)])
    print(f"Total road length: {total_length/1000:.2f} km")
    
    return G

def add_edge_speeds(G):
    """
    Add speed limits to edges based on road type
    Uses standard speeds when OSM data doesn't have speed info
    """
    
    # Default speeds (km/h) by road type
    speed_limits = {
        'motorway': 100,
        'motorway_link': 80,
        'trunk': 80,
        'trunk_link': 60,
        'primary': 65,
        'primary_link': 50,
        'secondary': 55,
        'secondary_link': 40,
        'tertiary': 50,
        'tertiary_link': 40,
        'residential': 40,
        'living_street': 30,
        'unclassified': 40,
        'service': 30,
        'road': 40
    }
    
    for u, v, key, data in G.edges(keys=True, data=True):
        # Get road type
        highway_type = data.get('highway', 'residential')
        
        # Handle lists (sometimes OSM returns multiple types)
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        
        # Get speed limit or use default
        if 'maxspeed' in data:
            try:
                speed = float(data['maxspeed'])
            except (ValueError, TypeError):
                speed = speed_limits.get(highway_type, 40)
        else:
            speed = speed_limits.get(highway_type, 40)
        
        # Add to edge data
        data['speed_limit'] = speed
        
        # Calculate free-flow travel time (in seconds)
        length_km = data['length'] / 1000
        time_hours = length_km / speed
        data['free_flow_time'] = time_hours * 3600  # convert to seconds
    
    return G

def save_network(G, base_path='data/raw'):
    """
    Save network in multiple formats
    """
    import os
    os.makedirs(base_path, exist_ok=True)
    
    # Save as GraphML (preserves all data)
    graphml_path = f"{base_path}/sf_network.graphml"
    ox.save_graphml(G, graphml_path)
    print(f"✓ Saved GraphML: {graphml_path}")
    
    # Save as pickle (fastest to load)
    pickle_path = f"{base_path}/sf_network.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"✓ Saved Pickle: {pickle_path}")
    
    # Save basic statistics as JSON
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'total_length_km': sum([data['length'] for u, v, data in G.edges(data=True)]) / 1000
    }
    stats_path = f"{base_path}/network_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved Stats: {stats_path}")
    
    return graphml_path

def visualize_network(G, save_path='results/network_preview.html'):
    """
    Create an interactive map preview
    """
    import os
    os.makedirs('results', exist_ok=True)
    
    # Create folium map
    m = ox.plot_graph_folium(G, popup_attribute='name', edge_width=2)
    
    # Save
    m.save(save_path)
    print(f"✓ Saved interactive map: {save_path}")
    print(f"  Open in browser to view")
    
    return save_path

def main():
    """
    Main execution
    """
    print("="*60)
    print("San Francisco Downtown Road Network Download")
    print("="*60)
    
    # Download network
    G = download_sf_network(method='bbox')
    
    # Add speed limits
    print("\nAdding speed limits to edges...")
    G = add_edge_speeds(G)
    
    # Save network
    print("\nSaving network...")
    save_network(G)
    
    # Visualize
    print("\nCreating visualization...")
    visualize_network(G)
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)

if __name__ == "__main__":
    main()