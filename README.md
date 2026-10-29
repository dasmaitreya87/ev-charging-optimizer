![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
# EV Charging Station Routing Optimization

A smart routing system that helps electric vehicle drivers find a feasible charging stop and compare route options across a real road network.

<img width="1880" height="911" alt="image" src="https://github.com/user-attachments/assets/e4ac2dbb-74bb-4ee2-8267-ac9f8cd02482" />



## 🎯 What This Project Does

This project optimizes EV charging station routes by considering:
- **Distance** - Minimize travel distance
- **Real-time Traffic** - Account for traffic conditions
- **Station Occupancy** - Avoid crowded stations

**Routing approaches:**
- **ACO** - Ant Colony Optimization (bio-inspired)
- **PSO** - Particle Swarm Optimization (swarm intelligence)
- **Interactive route planner** - Fast distance- and preference-weighted shortest-path searches on the road graph.

##  Features

-  Real road network from OpenStreetMap (San Francisco)
-  Live charging station data from Open Charge Map API
-  Google Maps API integration for real traffic data
-  Interactive web interface for route visualization
-  Side-by-side route comparison

##  Quick Start

### Installation

Use Python 3.11 on Windows. From PowerShell, open the project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Put your Google Maps key in the root `.env` file:

```env
GOOGLE_MAPS_API_KEY=your_actual_api_key
```

The `.env` file is ignored by Git. If it does not exist, copy `.env.example` to `.env` and replace the placeholder.

### Prepare data and run the web app

Run these commands from the project root in PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe src\download_network.py
.\.venv\Scripts\python.exe src\charging_stations.py
.\.venv\Scripts\python.exe src\google_traffic_api.py --sample-edges 100
.\.venv\Scripts\python.exe run_web_server.py
```

The station downloader tries Open Charge Map first. If that API is unavailable, it uses valid saved station data or generates demo stations on the road network. The traffic command samples 100 edges to limit Google API usage. Before using live traffic, enable the **Directions API (Legacy)** for your Google Cloud project and ensure billing is configured. This project currently calls that API. If live requests are denied, it writes free-flow fallback traffic data so the web app can still run; the traffic summary reports the API success rate.

After the server starts, open **http://127.0.0.1:5000**. In the **Trip locations** form, enter the start and final-destination latitude/longitude, or use **Set start** and **Set destination** and click the map; map selections fill in the coordinate fields, and either pin can be dragged to fine-tune its location. Enter the vehicle's current battery, reserve, estimated full-charge range, and a ride preference:

- **Balanced** weighs distance, travel time, and station wait.
- **Fastest** prioritizes travel time.
- **Shortest** prioritizes distance.
- **Avoid busy stations** prioritizes estimated wait time.

Select **Find Best Route** to find charging stations that can be reached before using your reserve and can then reach the final destination after charging. The planner runs one forward and one reverse graph search to check candidates, then draws a shortest-distance route (red) and a distance/time-weighted route for your ride preference (blue). These two fast graph-search routes are not ACO/PSO: red is shortest distance to the chosen station, and blue minimizes weighted distance/time to that same station; both use the shortest-distance onward leg. The **Include ACO & PSO comparison** option is enabled by default and runs the actual ACO (8 ants × 8 iterations) and PSO (8 particles × 5 iterations) implementations; any valid optimizer routes appear as orange and purple dashed lines. Turn it off for the fastest graph-only result. The result explains why the recommended station has the lowest weighted score, shows the score terms, and compares nearby feasible stations. Station scores use distance weight × complete trip km + time weight × (estimated drive + wait minutes) + occupancy weight × estimated wait minutes; stations that fail either battery reserve check are excluded. The estimate assumes the vehicle charges to 100% at the selected station, while preserving the entered reserve on both legs.

##  Project Structure

```
ev_charging_routing/
├── src/
│   ├── aco_algorithm.py          # ACO implementation
│   ├── pso_algorithm.py          # PSO implementation
│   ├── web_server.py             # Flask web app
│   ├── google_traffic_api.py     # Traffic data integration
│   └── main.py                   # Data pipeline
├── data/
│   ├── raw/                      # Road network data
│   └── processed/                # Charging stations & traffic
├── run_web_server.py             # Web server launcher
└── requirements.txt              # Dependencies
```

## How It Works

### Ant Colony Optimization (ACO)
- Virtual ants explore routes to charging stations
- Better routes receive more pheromone
- Future ants prefer high-pheromone paths
- Converges to optimal solution

### Particle Swarm Optimization (PSO)
- Particles move through solution space
- Influenced by personal best and global best
- Swarm converges toward optimal routes

### Cost Function
```
Total Cost = 0.3 × Distance + 0.4 × Time + 0.3 × Waiting Time
```

##  Usage Example

```python
from src.aco_algorithm import ACO_EV_Router
import pickle, pandas as pd

# Load data
with open('data/raw/sf_network.pkl', 'rb') as f:
    G = pickle.load(f)
stations_df = pd.read_csv('data/processed/charging_stations.csv')
traffic_df = pd.read_csv('data/processed/traffic_data_24h.csv')

# Run ACO
aco = ACO_EV_Router(
    graph=G,
    stations_df=stations_df,
    traffic_df=traffic_df,
    start_node=12345,
    current_hour=17,
    battery_level=0.2
)

best_path, best_cost, _ = aco.run()
print(f"Best route cost: {best_cost:.2f}")
```

##  Configuration

Customize algorithm parameters in `src/aco_algorithm.py` or `src/pso_algorithm.py`:

```python
# ACO parameters
n_ants=20           # Number of ants
n_iterations=100    # Iterations
alpha=1.5           # Pheromone importance
beta=2.0            # Heuristic importance

# PSO parameters
n_particles=30      # Swarm size
w=0.7               # Inertia weight
c1=1.5, c2=1.5     # Cognitive/social coefficients
```

##  Technologies Used

- **OSMnx** - Road network data from OpenStreetMap
- **NetworkX** - Graph algorithms and pathfinding
- **Flask** - Web interface
- **Folium** - Interactive maps
- **Google Maps API** - Real-time traffic (optional)
- **NumPy/Pandas** - Data processing
