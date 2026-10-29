"""
Flask web server for integrated route selection and visualization
Provides a Google Maps-like experience where users can select points
and see routes on the same page
"""

import pickle
import pandas as pd
import numpy as np
import networkx as nx
import folium
import json
import os
import subprocess
import sys
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file
import threading
import time

# find_nearest_node function (moved from select_route_points.py)
def find_nearest_node(G, lat, lon):
    """
    Find the nearest network node to given coordinates
    
    Parameters:
    -----------
    G : networkx.MultiDiGraph
        Road network
    lat : float
        Latitude
    lon : float
        Longitude
    
    Returns:
    --------
    int or None
        Nearest node ID, or None if not found
    """
    try:
        import osmnx
        # Use osmnx's nearest_nodes for accurate finding
        nearest_node = osmnx.distance.nearest_nodes(G, lon, lat)
        return nearest_node
    except Exception as e:
        # Fallback: manual search
        min_dist = float('inf')
        nearest_node = None
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_lat = node_data.get('y', 0)
            node_lon = node_data.get('x', 0)
            
            # Calculate distance (Haversine approximation)
            dist = np.sqrt((node_lat - lat)**2 + (node_lon - lon)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        return nearest_node

app = Flask(__name__)

# Global variables for data (loaded once at startup)
G = None
stations_df = None
traffic_df = None
original_stations_df = None

ROUTE_PREFERENCES = {
    'balanced': {'label': 'Balanced ride', 'weights': (0.3, 0.4, 0.3)},
    'fastest': {'label': 'Fastest ride', 'weights': (0.15, 0.65, 0.2)},
    'shortest': {'label': 'Shortest ride', 'weights': (0.65, 0.2, 0.15)},
    'low_wait': {'label': 'Avoid busy stations', 'weights': (0.2, 0.25, 0.55)}
}

def load_data():
    """Load network and station data"""
    global G, stations_df, traffic_df, original_stations_df
    
    print("Loading data...")
    try:
        with open('data/raw/sf_network.pkl', 'rb') as f:
            G = pickle.load(f)
        print(f"✓ Loaded network: {G.number_of_nodes():,} nodes")
    except FileNotFoundError:
        print("ERROR: Could not find data/raw/sf_network.pkl")
        return False
    
    try:
        stations_df = pd.read_csv('data/processed/charging_stations.csv')
        original_stations_df = stations_df.copy()
        print(f"✓ Loaded {len(stations_df)} charging stations")
    except FileNotFoundError:
        print("ERROR: Could not find data/processed/charging_stations.csv")
        return False
    
    try:
        traffic_df = pd.read_csv('data/processed/traffic_data_24h.csv')
        print(f"✓ Loaded traffic data")
    except FileNotFoundError:
        print("ERROR: Could not find data/processed/traffic_data_24h.csv")
        return False
    
    return True

def create_selection_map():
    """Create the interactive selection map"""
    if G is None or stations_df is None:
        return None
    
    # Calculate map center
    if len(stations_df) > 0:
        center_lat = stations_df['latitude'].mean()
        center_lon = stations_df['longitude'].mean()
    else:
        first_node = list(G.nodes())[0]
        center_lat = G.nodes[first_node]['y']
        center_lon = G.nodes[first_node]['x']
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add fullscreen button
    from folium import plugins
    plugins.Fullscreen().add_to(m)
    
    # Add all charging stations as markers (for reference only, not clickable)
    for idx, station in stations_df.iterrows():
        station_lat = station['latitude']
        station_lon = station['longitude']
        station_name = station['name']
        
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>{station_name}</b><br>
            Ports: {station['num_ports']}
        </div>
        """
        
        folium.Marker(
            [station_lat, station_lon],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=station_name,
            icon=folium.Icon(color='blue', icon='bolt', prefix='fa')
        ).add_to(m)
    
    # JavaScript for map interaction
    from branca.element import Element
    
    js_code = """
    <script>
    (function() {
        var startMarker = null;
        var startCoords = null;
        var destinationMarker = null;
        var destinationCoords = null;
        var theMap = null;
        var routeLayers = []; // Store route layers to remove them later
        var mapMode = 'navigate';
        var coordinateListenersAttached = false;
        
        function getMap() {
            if (theMap) return theMap;
            if (typeof window !== 'undefined') {
                for (var prop in window) {
                    if (window[prop] && window[prop]._container && window[prop].on) {
                        theMap = window[prop];
                        return theMap;
                    }
                }
            }
            var containers = document.querySelectorAll('.folium-map, [id^="map"]');
            for (var i = 0; i < containers.length; i++) {
                var container = containers[i];
                if (container._leaflet_id) {
                    var id = container._leaflet_id;
                    if (L._layers && L._layers[id]) {
                        theMap = L._layers[id];
                        return theMap;
                    }
                }
            }
            if (typeof L !== 'undefined' && L._layers) {
                for (var key in L._layers) {
                    var layer = L._layers[key];
                    if (layer && layer._container && layer.on && layer.addLayer) {
                        theMap = layer;
                        return theMap;
                    }
                }
            }
            return null;
        }

        window.setMapMode = function(mode) {
            mapMode = mode;
            var map = getMap();
            var navigateButton = document.getElementById('navigate-mode');
            var selectButton = document.getElementById('select-mode');
            var destinationButton = document.getElementById('destination-mode');
            if (navigateButton) {
                navigateButton.classList.toggle('active', mode === 'navigate');
                navigateButton.setAttribute('aria-pressed', mode === 'navigate');
            }
            if (selectButton) {
                selectButton.classList.toggle('active', mode === 'select');
                selectButton.setAttribute('aria-pressed', mode === 'select');
            }
            if (destinationButton) {
                destinationButton.classList.toggle('active', mode === 'destination');
                destinationButton.setAttribute('aria-pressed', mode === 'destination');
            }
            if (map) {
                map.getContainer().style.cursor = mode === 'select' ? 'crosshair' : 'grab';
                if (mode === 'destination') map.getContainer().style.cursor = 'copy';
            }
        };
        
        window.selectStart = function(lat, lon) {
            var map = getMap();
            if (!map) {
                showFormError('The map is still loading. Please wait a moment and try again.');
                return;
            }
            if (startMarker) map.removeLayer(startMarker);
            var greenIcon = new L.Icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            });
            startMarker = L.marker([lat, lon], {icon: greenIcon, draggable: true}).addTo(map);
            startMarker.bindPopup('<b>Start Point</b><br>Lat: ' + lat.toFixed(6) + '<br>Lon: ' + lon.toFixed(6) + 
                                 '<br><button onclick="window.clearStart()" style="margin:5px; padding:5px 10px;">Clear</button>');
            startMarker.on('dragend', function(e) {
                var pos = e.target.getLatLng();
                startCoords = [pos.lat, pos.lng];
                setCoordinateInputs('start', pos.lat, pos.lng);
                updateStatus();
            });
            startCoords = [lat, lon];
            setCoordinateInputs('start', lat, lon);
            updateStatus();
            window.setMapMode('navigate');
        };

        window.selectDestination = function(lat, lon) {
            var map = getMap();
            if (!map) {
                showFormError('The map is still loading. Please wait a moment and try again.');
                return;
            }
            if (destinationMarker) map.removeLayer(destinationMarker);
            destinationMarker = L.marker([lat, lon], {draggable: true}).addTo(map);
            destinationMarker.setIcon(new L.Icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-violet.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            }));
            destinationMarker.bindPopup('<b>Final Destination</b><br>Drag to adjust your destination.');
            destinationMarker.on('dragend', function(e) {
                var pos = e.target.getLatLng();
                destinationCoords = [pos.lat, pos.lng];
                setCoordinateInputs('destination', pos.lat, pos.lng);
                updateStatus();
            });
            destinationCoords = [lat, lon];
            setCoordinateInputs('destination', lat, lon);
            updateStatus();
            window.setMapMode('navigate');
        };

        function setCoordinateInputs(point, lat, lon) {
            var prefix = point === 'start' ? 'start' : 'destination';
            var latitudeInput = document.getElementById(prefix + '-latitude');
            var longitudeInput = document.getElementById(prefix + '-longitude');
            if (latitudeInput) latitudeInput.value = lat.toFixed(6);
            if (longitudeInput) longitudeInput.value = lon.toFixed(6);
        }

        function showFormError(message) {
            var statusDiv = document.getElementById('selection-status');
            if (statusDiv) {
                statusDiv.innerHTML = '<h3 style="color:#b91c1c;">Please check your trip details</h3><p>' + message + '</p>';
            }
        }

        function escapeHtml(value) {
            return String(value).replace(/[&<>"']/g, function(character) {
                return {
                    '&': '&amp;', '<': '&lt;', '>': '&gt;',
                    '"': '&quot;', "'": '&#39;'
                }[character];
            });
        }

        window.clearStart = function() {
            var map = getMap();
            if (startMarker && map) {
                map.removeLayer(startMarker);
                startMarker = null;
                startCoords = null;
                updateStatus();
            }
        };

        window.clearDestination = function() {
            var map = getMap();
            if (destinationMarker && map) {
                map.removeLayer(destinationMarker);
                destinationMarker = null;
                destinationCoords = null;
                updateStatus();
            }
        };
        
        function updateStatus() {
            var div = document.getElementById('selection-status');
            var btn = document.getElementById('save-button');
            if (!div) return;
            var html = '<h3>Selection Status:</h3>';
            if (startCoords) {
                html += '<p style="color:green;">✓ Start: ' + startCoords[0].toFixed(6) + ', ' + startCoords[1].toFixed(6) + '</p>';
            } else {
                html += '<p style="color:red;">✗ Start: Enter coordinates below or choose Set start and click the map</p>';
            }
            if (destinationCoords) {
                html += '<p style="color:#7c3aed;">✓ Final destination: ' + destinationCoords[0].toFixed(6) + ', ' + destinationCoords[1].toFixed(6) +
                    ' <button type="button" onclick="window.clearDestination()" class="clear-point">Clear</button></p>';
            } else {
                html += '<p style="color:#7c3aed;">✗ Final destination: Enter coordinates below or choose Set destination and click the map</p>';
            }
            if (startCoords && destinationCoords) {
                html += '<p style="color:#0f766e;">A reachable charging station will be selected between these points.</p>';
            }
            div.innerHTML = html;
            if (btn) {
                if (startCoords && destinationCoords) {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                    btn.style.cursor = 'pointer';
                } else {
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                    btn.style.cursor = 'not-allowed';
                }
            }
        }

        function attachCoordinateInputListeners() {
            if (coordinateListenersAttached) return;
            coordinateListenersAttached = true;
            ['start', 'destination'].forEach(function(point) {
                ['latitude', 'longitude'].forEach(function(axis) {
                    document.getElementById(point + '-' + axis).addEventListener('input', function() {
                        var latitude = document.getElementById(point + '-latitude').value.trim();
                        var longitude = document.getElementById(point + '-longitude').value.trim();
                        var lat = Number(latitude);
                        var lon = Number(longitude);
                        if (latitude && longitude && Number.isFinite(lat) && Number.isFinite(lon) &&
                            lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {
                            if (point === 'start') startCoords = [lat, lon];
                            else destinationCoords = [lat, lon];
                        } else if (point === 'start') {
                            startCoords = null;
                        } else {
                            destinationCoords = null;
                        }
                        updateStatus();
                    });
                });
            });
        }

        window.saveSelection = function() {
            var values = {};
            var coordinateInputsEmpty = false;
            ['start', 'destination'].forEach(function(point) {
                var latitudeValue = document.getElementById(point + '-latitude').value.trim();
                var longitudeValue = document.getElementById(point + '-longitude').value.trim();
                if (!latitudeValue || !longitudeValue) coordinateInputsEmpty = true;
                values[point + 'Lat'] = Number(latitudeValue);
                values[point + 'Lon'] = Number(longitudeValue);
            });
            if (coordinateInputsEmpty || !Number.isFinite(values.startLat) || !Number.isFinite(values.startLon) ||
                !Number.isFinite(values.destinationLat) || !Number.isFinite(values.destinationLon)) {
                showFormError('Enter the latitude and longitude for both your start and final destination, or select both points on the map.');
                return;
            }
            if (Math.abs(values.startLat) > 90 || Math.abs(values.destinationLat) > 90 ||
                Math.abs(values.startLon) > 180 || Math.abs(values.destinationLon) > 180) {
                showFormError('Latitude must be between -90 and 90; longitude must be between -180 and 180.');
                return;
            }
            if (Math.abs(values.startLat - values.destinationLat) < 1e-8 &&
                Math.abs(values.startLon - values.destinationLon) < 1e-8) {
                showFormError('Choose a final destination different from your start point.');
                return;
            }

            if (!startCoords || startCoords[0] !== values.startLat || startCoords[1] !== values.startLon) {
                window.selectStart(values.startLat, values.startLon);
            }
            if (!destinationCoords || destinationCoords[0] !== values.destinationLat || destinationCoords[1] !== values.destinationLon) {
                window.selectDestination(values.destinationLat, values.destinationLon);
            }
            
            var data = {
                start_lat: values.startLat,
                start_lon: values.startLon,
                destination_lat: values.destinationLat,
                destination_lon: values.destinationLon,
                battery_percent: Number(document.getElementById('battery-percent').value),
                reserve_percent: Number(document.getElementById('reserve-percent').value),
                max_range_km: Number(document.getElementById('max-range-km').value),
                route_preference: document.getElementById('route-preference').value,
                run_aco_pso: document.getElementById('run-aco-pso').checked
            };
            if (!Number.isFinite(data.battery_percent) || !Number.isFinite(data.reserve_percent) ||
                !Number.isFinite(data.max_range_km) || data.battery_percent < 1 || data.battery_percent > 100 ||
                data.reserve_percent < 0 || data.reserve_percent >= data.battery_percent ||
                data.max_range_km < 20 || data.max_range_km > 1000) {
                showFormError('Current charge must be 1–100%, reserve must be 0% and below your current charge, and full range must be 20–1000 km.');
                return;
            }
            
            // Show loading message
            var statusDiv = document.getElementById('selection-status');
            statusDiv.innerHTML = '<div class="searching"><span class="search-spinner"></span><div><h3>Finding a charging stop</h3><p>Checking reachable stations and comparing routes' +
                (data.run_aco_pso ? ', including bounded ACO and PSO runs' : '') + '.</p></div></div>';
            var btn = document.getElementById('save-button');
            if (btn) {
                btn.disabled = true;
                btn.textContent = 'Calculating Routes...';
            }
            ['battery-percent', 'reserve-percent', 'max-range-km', 'route-preference', 'run-aco-pso'].forEach(function(id) {
                document.getElementById(id).disabled = true;
            });
            ['start-latitude', 'start-longitude', 'destination-latitude', 'destination-longitude'].forEach(function(id) {
                document.getElementById(id).disabled = true;
            });
            
            // Send to server
            fetch('/calculate_routes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json().then(function(result) {
                if (!response.ok) {
                    throw new Error(result.error || 'Route calculation failed');
                }
                return result;
            }))
            .then(result => {
                if (result.success) {
                    // Remove old route layers
                    var map = getMap();
                    routeLayers.forEach(function(layer) {
                        if (map && map.hasLayer(layer)) {
                            map.removeLayer(layer);
                        }
                    });
                    routeLayers = [];

                    // Add routes to map
                    if (result.routes) {
                        // Add start marker using user's selected coordinates (only once)
                        if (result.start_lat && result.start_lon) {
                            var startIcon = L.icon({
                                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
                                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                                iconSize: [25, 41],
                                iconAnchor: [12, 41]
                            });
                            var routeStartMarker = L.marker([result.start_lat, result.start_lon], {icon: startIcon}).addTo(map);
                            routeStartMarker.bindPopup('<b>Start Point</b><br>Your selected location');
                            routeLayers.push(routeStartMarker);
                        }
                        
                        // Add destination marker using user's selected coordinates (only once)
                        if (result.station_lat !== null && result.station_lon !== null) {
                            var destIcon = L.icon({
                                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
                                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                                iconSize: [25, 41],
                                iconAnchor: [12, 41]
                            });
                            var routeDestMarker = L.marker([result.station_lat, result.station_lon], {icon: destIcon}).addTo(map);
                            var destName = result.station_name || 'Charging Station';
                            routeDestMarker.bindPopup('<b>Charging Stop: ' + destName + '</b>');
                            routeLayers.push(routeDestMarker);
                        }
                        if (result.final_destination_lat !== null && result.final_destination_lon !== null) {
                            var finalIcon = L.icon({
                                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-violet.png',
                                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                                iconSize: [25, 41],
                                iconAnchor: [12, 41]
                            });
                            var finalMarker = L.marker([result.final_destination_lat, result.final_destination_lon], {icon: finalIcon}).addTo(map);
                            finalMarker.bindPopup('<b>Final Destination</b>');
                            routeLayers.push(finalMarker);
                        }
                        
                        // Add route polylines
                        result.routes.forEach(function(route) {
                            if (route.coords && route.coords.length >= 2) {
                                var color = route.color || 'red';
                                var weight = route.weight || 5;
                                var polyline = L.polyline(route.coords, {
                                    color: color,
                                    weight: weight,
                                    opacity: 0.85,
                                    dashArray: route.dash_array || null
                                }).addTo(map);
                                routeLayers.push(polyline);
                                
                                // Add popup
                                var popupText = '<b>' + route.name + '</b><br>Cost: ' +
                                    (route.cost === null ? 'N/A' : route.cost.toFixed(2));
                                polyline.bindPopup(popupText);
                            }
                        });
                        if (result.routes.length) {
                            var bounds = L.featureGroup(routeLayers).getBounds();
                            if (bounds.isValid()) map.fitBounds(bounds, {paddingTopLeft: [24, 24], paddingBottomRight: [390, 24], maxZoom: 16});
                        }
                    }
                    
                    // Update status
                    var destName = result.destination_name || 'Recommended station';
                    var routeSummary = result.routes.map(function(route) {
                        var color = route.color || '#475569';
                        return '<span style="color:' + color + '">■</span> ' + escapeHtml(route.name) +
                            ' · score ' + route.cost.toFixed(2);
                    }).join('<br>');
                    var comparisonRows = result.station_comparison.map(function(candidate) {
                        var selectedClass = candidate.selected ? ' class="selected-candidate"' : '';
                        return '<tr' + selectedClass + '><td>' + escapeHtml(candidate.name) +
                            (candidate.selected ? ' <b>(selected)</b>' : '') + '<br><span class="result-note">' +
                            (candidate.selected ? 'Recommended' : candidate.distance_from_selected_km.toFixed(1) + ' km from selected stop') +
                            ' · score rank #' + candidate.score_rank + '</span></td><td>' +
                            candidate.distance_km.toFixed(1) + ' km</td><td>' + candidate.driving_minutes.toFixed(1) +
                            ' min</td><td>' + candidate.wait_minutes.toFixed(1) + ' min</td><td>' +
                            candidate.station_arrival_percent.toFixed(1) + '% / ' + candidate.final_arrival_percent.toFixed(1) + '%</td><td><span class="component-score">D ' +
                            candidate.distance_score.toFixed(2) + ' + T ' + candidate.time_score.toFixed(2) + ' + W ' +
                            candidate.occupancy_score.toFixed(2) + '</span></td><td><b>' + candidate.score.toFixed(2) + '</b></td></tr>';
                    }).join('');
                    var optimizerRows = ['ACO', 'PSO'].map(function(name) {
                        var optimizer = result.optimizer_results[name];
                        return '<div class="optimizer-result"><b>' + name + '</b><span>' +
                            escapeHtml(optimizer.message) + '</span></div>';
                    }).join('');
                    var scoreWeights = result.station_score_weights;
                    statusDiv.innerHTML = '<h3 style="color:#0f766e;">✓ Recommended stop found</h3>' +
                        '<p><b>' + destName + '</b> · ' + result.feasible_station_count + ' reachable stations checked</p>' +
                        '<div class="result-metrics">' +
                            '<div><span>Trip distance</span><b>' + result.distance_km.toFixed(1) + ' km</b></div>' +
                            '<div><span>At charging stop</span><b>' + result.station_arrival_percent.toFixed(1) + '%</b></div>' +
                            '<div><span>At final destination</span><b>' + result.final_arrival_percent.toFixed(1) + '%</b></div>' +
                            '<div><span>Ride preference</span><b>' + result.route_preference_label + '</b></div>' +
                        '</div>' +
                        '<p class="result-note">Charging estimate assumes a full charge at the stop and keeps your ' + result.reserve_percent + '% reserve.</p>' +
                        '<div class="route-score-list">' + routeSummary + '</div>' +
                        '<details class="result-details" open><summary>Why this station?</summary>' +
                        '<p>' + escapeHtml(result.station_reason) + '</p>' +
                        '<p class="result-note">Weights used: distance ' + (scoreWeights.distance * 100).toFixed(0) +
                        '%, time ' + (scoreWeights.time * 100).toFixed(0) + '%, occupancy/wait ' +
                        (scoreWeights.occupancy * 100).toFixed(0) + '%.</p>' +
                        '<p class="result-note">Station score = distance weight × trip km + time weight × (drive + estimated wait minutes) + occupancy weight × estimated wait minutes. D = distance points, T = time points, W = station-wait/occupancy points. Only stations that preserve your battery reserve on both legs are included.</p>' +
                        '<div class="comparison-scroll"><table class="station-comparison"><thead><tr><th>Nearby feasible station</th><th>Trip</th><th>Drive</th><th>Wait</th><th>Charge at stop / destination</th><th>Score parts (D + T + W)</th><th>Total score</th></tr></thead><tbody>' +
                        comparisonRows + '</tbody></table></div></details>' +
                        '<section class="result-details"><h4 class="optimizer-heading">ACO &amp; PSO results</h4>' +
                        optimizerRows + '<details class="optimizer-explanation"><summary>How these algorithms work</summary>' +
                        '<p class="result-note">' + escapeHtml(result.optimizer_explanation) + '</p></details></section>';

                    if (btn) {
                        btn.disabled = false;
                        btn.textContent = 'Find Best Route';
                    }
                    ['battery-percent', 'reserve-percent', 'max-range-km', 'route-preference', 'run-aco-pso'].forEach(function(id) {
                        document.getElementById(id).disabled = false;
                    });
                    ['start-latitude', 'start-longitude', 'destination-latitude', 'destination-longitude'].forEach(function(id) {
                        document.getElementById(id).disabled = false;
                    });
                } else {
                    throw new Error(result.error || 'Unknown error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                var statusDiv = document.getElementById('selection-status');
                statusDiv.innerHTML = '<h3 style="color:#b91c1c;">Route search failed</h3><p>' + error.message + '</p>';
                ['battery-percent', 'reserve-percent', 'max-range-km', 'route-preference', 'run-aco-pso'].forEach(function(id) {
                    document.getElementById(id).disabled = false;
                });
                ['start-latitude', 'start-longitude', 'destination-latitude', 'destination-longitude'].forEach(function(id) {
                    document.getElementById(id).disabled = false;
                });
                var btn = document.getElementById('save-button');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = 'Find Best Route';
                }
            });
        };

        function setupMapClick() {
            attachCoordinateInputListeners();
            var map = getMap();
            if (map) {
                map.on('click', function(e) {
                    if (mapMode === 'select') {
                        window.selectStart(e.latlng.lat, e.latlng.lng);
                    } else if (mapMode === 'destination') {
                        window.selectDestination(e.latlng.lat, e.latlng.lng);
                    }
                });
                map.on('mousedown', function() {
                    if (mapMode === 'navigate') {
                        map.getContainer().style.cursor = 'grabbing';
                    }
                });
                map.on('mouseup', function() {
                    window.setMapMode(mapMode);
                });
                window.setMapMode('navigate');
                updateStatus();
            } else {
                setTimeout(setupMapClick, 300);
            }
        }
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(setupMapClick, 500);
            });
        } else {
            setTimeout(setupMapClick, 500);
        }
    })();
    </script>
    """
    
    m.get_root().html.add_child(Element(js_code))
    
    # Add selection panel
    selection_html = """
    <style>
        .route-panel {
            position: fixed; top: 14px; right: 14px; width: min(400px, calc(100vw - 28px));
            max-height: calc(100vh - 28px); overflow-y: auto;
            box-sizing: border-box; z-index: 9999; padding: 20px;
            color: #142033; background: rgba(255,255,255,.96);
            border: 1px solid rgba(203,213,225,.9); border-radius: 20px;
            box-shadow: 0 18px 55px rgba(15,23,42,.22);
            backdrop-filter: blur(14px);
            font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
        }
        .route-panel-header { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
        .brand-mark { display: grid; place-items: center; flex: 0 0 42px; height: 42px; border-radius: 14px; color: #fff; background: linear-gradient(145deg,#2563eb,#0f766e); font-size: 22px; box-shadow: 0 6px 14px rgba(37,99,235,.22); }
        .route-panel h2 { margin: 0; font-size: 21px; letter-spacing: -.03em; }
        .route-panel .subtitle { margin: 0 0 16px 54px; color: #64748b; font-size: 13px; }
        .section-title { margin: 0 0 8px; color: #334155; font-size: 13px; font-weight: 750; }
        .map-mode-control { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin-bottom: 12px; }
        .map-mode-button {
            padding: 10px 5px; border: 1px solid #d8e0eb; border-radius: 11px;
            color: #334155; background: #f8fafc; font: inherit; font-size: 12px; line-height: 1.3; font-weight: 600;
            cursor: pointer; transition: transform .15s, background .15s, color .15s, border-color .15s;
        }
        .map-mode-button:hover { transform: translateY(-1px); border-color: #93c5fd; }
        .map-mode-button.active { color: #fff; background: #1d4ed8; border-color: #1d4ed8; box-shadow: 0 4px 10px rgba(29,78,216,.2); }
        .map-mode-button:focus-visible, .route-action:focus-visible { outline: 3px solid #93c5fd; outline-offset: 2px; }
        #selection-status { padding: 12px; margin-top: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 13px; }
        #selection-status h3 { margin: 0 0 6px; font-size: 14px; }
        #selection-status p { margin: 5px 0; overflow-wrap: anywhere; }
        .route-action {
            width: 100%; padding: 13px 14px; margin-top: 12px; border: 0;
            border-radius: 12px; color: white; background: linear-gradient(110deg,#0f766e,#0d9488);
            font: inherit; font-weight: 750; cursor: pointer; box-shadow: 0 6px 14px rgba(15,118,110,.18);
            transition: transform .15s, box-shadow .15s, opacity .15s;
        }
        .route-action:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 9px 18px rgba(15,118,110,.25); }
        .route-action:disabled { opacity: .5; cursor: not-allowed; }
        .route-legend { margin: 12px 0 0; color: #475569; font-size: 12px; }
        .route-instructions { margin: 12px 0 0; padding-left: 19px; color: #64748b; font-size: 12px; }
        .route-instructions li { margin: 4px 0; }
        .vehicle-settings { margin: 12px 0 0; padding: 12px; border: 1px solid #e2e8f0; border-radius: 13px; background: #fff; }
        .vehicle-settings summary { cursor: pointer; color: #334155; font-weight: 750; }
        .settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
        .settings-grid label { display: grid; gap: 4px; color: #475569; font-size: 12px; font-weight: 600; }
        .settings-grid input, .settings-grid select { min-width: 0; padding: 9px; border: 1px solid #cbd5e1; border-radius: 9px; color: #172033; background: #fff; font: inherit; }
        .settings-grid input:focus, .settings-grid select:focus, .location-grid input:focus { outline: 3px solid #bfdbfe; border-color: #3b82f6; }
        .location-settings { margin: 12px 0 0; padding: 12px; border: 1px solid #e2e8f0; border-radius: 13px; background: #fff; }
        .location-settings h3 { margin: 0; font-size: 14px; }
        .location-settings .setting-hint { display: block; margin-top: 4px; }
        .location-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
        .location-grid fieldset { min-width: 0; margin: 0; padding: 9px; border: 1px solid #dce3ed; border-radius: 10px; background: #fbfdff; }
        .location-grid legend { padding: 0 4px; color: #475569; font-size: 12px; font-weight: 750; }
        .location-grid label { display: grid; gap: 3px; margin-top: 6px; color: #475569; font-size: 11px; font-weight: 600; }
        .location-grid input { box-sizing: border-box; width: 100%; min-width: 0; padding: 8px; border: 1px solid #cbd5e1; border-radius: 8px; color: #172033; background: #fff; font: inherit; }
        .setting-hint { grid-column: 1 / -1; margin: 0; color: #64748b; font-size: 11px; }
        .clear-point { margin-left: 5px; padding: 2px 6px; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; cursor: pointer; }
        .searching { display: flex; align-items: flex-start; gap: 10px; }
        .searching h3 { margin: 0 0 4px; }
        .searching p { margin: 4px 0; }
        .search-spinner { flex: 0 0 22px; width: 22px; height: 22px; border: 3px solid #cbd5e1; border-top-color: #0f766e; border-radius: 50%; animation: route-spin .8s linear infinite; }
        .form-label { display: block; margin: 0 0 8px; color: #334155; font-size: 13px; font-weight: 750; }
        .route-panel .route-instructions { border-top: 1px solid #e2e8f0; padding-top: 10px; }
        .result-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }
        .result-metrics div { display: grid; gap: 2px; padding: 9px; border: 1px solid #e2e8f0; border-radius: 9px; background: #fff; }
        .result-metrics span { color: #64748b; font-size: 10px; }
        .result-metrics b { color: #142033; font-size: 13px; }
        .result-note { color: #64748b; font-size: 11px; }
        .route-score-list { margin: 8px 0; padding: 9px; border: 1px solid #e2e8f0; border-radius: 9px; background: #fff; font-size: 11px; line-height: 1.7; }
        .result-details { margin-top: 9px; padding: 10px; border: 1px solid #e2e8f0; border-radius: 10px; background: #fff; }
        .result-details summary { color: #334155; cursor: pointer; font-weight: 750; }
        .result-details p { font-size: 12px; }
        .comparison-scroll { overflow-x: auto; margin-top: 8px; }
        .station-comparison { width: 100%; border-collapse: collapse; font-size: 10px; white-space: nowrap; }
        .station-comparison th, .station-comparison td { padding: 6px 5px; border-bottom: 1px solid #e2e8f0; text-align: left; }
        .station-comparison th { color: #64748b; font-weight: 700; }
        .selected-candidate { background: #ecfdf5; }
        .component-score { color: #475569; font-size: 9px; }
        .optimizer-result { display: flex; justify-content: space-between; gap: 8px; padding: 7px 0; border-top: 1px solid #e2e8f0; font-size: 11px; }
        .optimizer-result span { color: #64748b; text-align: right; }
        .optimizer-heading { margin: 0 0 5px; color: #334155; font-size: 12px; }
        .optimizer-explanation { margin-top: 6px; }
        .algorithm-settings { margin-top: 12px; padding: 11px; border: 1px solid #bfdbfe; border-radius: 12px; background: #eff6ff; }
        .algorithm-settings label { display: flex; align-items: flex-start; gap: 8px; color: #1e3a8a; font-weight: 750; }
        .algorithm-settings input { margin-top: 3px; accent-color: #2563eb; }
        .algorithm-settings p { margin: 5px 0 0 23px; color: #475569; font-size: 11px; }
        .method-details { margin: 10px 0; padding: 10px 12px; border: 1px solid #e2e8f0; border-radius: 12px; background: #fff; }
        .method-details summary { cursor: pointer; color: #334155; font-size: 12px; font-weight: 750; }
        .method-details p { margin: 7px 0 0; color: #64748b; font-size: 11px; }
        @keyframes route-spin { to { transform: rotate(360deg); } }
        @media (max-width: 760px) {
            .route-panel { top: 8px; right: 8px; width: min(390px, calc(100vw - 16px)); max-height: calc(100vh - 16px); padding: 14px; }
            .route-panel .subtitle { margin-left: 0; }
            .map-mode-button { font-size: 11px; }
        }
    </style>
    <div class="route-panel">
        <div class="route-panel-header">
            <div class="brand-mark" aria-hidden="true">⚡</div>
            <h2>EV Route Planner</h2>
        </div>
        <p class="subtitle">Plan a charging stop and compare two efficient routes.</p>
        <div class="map-mode-control" role="group" aria-label="Map interaction mode">
            <button id="navigate-mode" class="map-mode-button active" type="button"
                    aria-pressed="true" onclick="window.setMapMode('navigate')">✋ Navigate</button>
            <button id="select-mode" class="map-mode-button" type="button"
                    aria-pressed="false" onclick="window.setMapMode('select')">＋ Set start</button>
            <button id="destination-mode" class="map-mode-button" type="button"
                    aria-pressed="false" onclick="window.setMapMode('destination')">◎ Set destination</button>
        </div>
        <section class="location-settings" aria-labelledby="location-settings-title">
            <h3 id="location-settings-title">Trip locations</h3>
            <p class="setting-hint">Enter coordinates below, or click Set start / Set destination and choose points on the map. Map selections fill these fields.</p>
            <div class="location-grid">
                <fieldset>
                    <legend>Start</legend>
                    <label>Latitude
                        <input id="start-latitude" type="number" min="-90" max="90" step="any" placeholder="e.g. 37.7749" autocomplete="off">
                    </label>
                    <label>Longitude
                        <input id="start-longitude" type="number" min="-180" max="180" step="any" placeholder="e.g. -122.4194" autocomplete="off">
                    </label>
                </fieldset>
                <fieldset>
                    <legend>Final destination</legend>
                    <label>Latitude
                        <input id="destination-latitude" type="number" min="-90" max="90" step="any" placeholder="e.g. 37.7901" autocomplete="off">
                    </label>
                    <label>Longitude
                        <input id="destination-longitude" type="number" min="-180" max="180" step="any" placeholder="e.g. -122.4052" autocomplete="off">
                    </label>
                </fieldset>
            </div>
        </section>
        <div id="selection-status">
            <p>Loading...</p>
        </div>
        <details class="vehicle-settings" open>
            <summary>Vehicle details & ride preference</summary>
            <div class="settings-grid">
                <label>Current charge (%)
                    <input id="battery-percent" type="number" min="1" max="100" step="1" value="70">
                </label>
                <label>Keep in reserve (%)
                    <input id="reserve-percent" type="number" min="0" max="50" step="1" value="15">
                </label>
                <label>Estimated full range (km)
                    <input id="max-range-km" type="number" min="20" max="1000" step="10" value="250">
                </label>
                <label>Ride preference
                    <select id="route-preference">
                        <option value="balanced">Balanced</option>
                        <option value="fastest">Fastest / time priority</option>
                        <option value="shortest">Shortest / distance priority</option>
                        <option value="low_wait">Avoid busy stations</option>
                    </select>
                </label>
                <p class="setting-hint">Enter your current dashboard charge. Your reserve is protected on both legs. The app assumes you charge to full at the selected station; full range is your estimated range at 100%.</p>
            </div>
        </details>
        <div class="algorithm-settings">
            <label><input id="run-aco-pso" type="checkbox" checked>
                <span>Include ACO &amp; PSO comparison</span>
            </label>
            <p>Runs the actual ACO and PSO implementations with a small iteration limit. Turn off for the fastest graph-only results.</p>
        </div>
        <button id="save-button" onclick="window.saveSelection()" 
                class="route-action" disabled>
            Find Best Route
        </button>
        <p class="route-legend"><b>Fast routes:</b> <span style="color:#dc2626">Red = shortest distance</span> ·
        <span style="color:#2563eb">Blue = ride preference</span><br>
        <b>Optimizers:</b> <span style="color:#ea580c">Orange dashed = ACO</span> ·
        <span style="color:#7c3aed">Purple dashed = PSO</span></p>
        <details class="method-details">
            <summary>How are routes calculated?</summary>
            <p><b>Red and blue are graph-search routes, not ACO/PSO.</b> Red uses the shortest-distance graph path to the recommended charger, then the shortest-distance onward path. Blue minimizes a distance/time weighted cost to the same charger and shares the onward leg.</p>
            <p><b>ACO:</b> Ant Colony Optimization repeatedly explores road paths using pheromone and route cost. <b>PSO:</b> Particle Swarm Optimization iteratively updates candidate paths toward better solutions. The optional bounded runs are separate dashed routes when they find a valid path.</p>
            <p>The recommended station minimizes the selected preference's weighted score among candidates feasible on both battery legs. The comparison table shows nearby feasible chargers, score rank, trip distance, estimated drive/wait and weighted score parts.</p>
        </details>
        <ol class="route-instructions">
            <li><b>Navigate</b> is the default: drag to pan and scroll to zoom.</li>
            <li>Enter start and destination coordinates above, or choose each point on the map. Drag either pin to adjust.</li>
            <li>Enter your battery details and ride preference, then select <b>Find Best Route</b>.</li>
        </ol>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(selection_html))
    
    return m

@app.route('/')
def index():
    """Main page with selection map"""
    if G is None or stations_df is None:
        return "Error: Data not loaded. Please check server logs."
    
    m = create_selection_map()
    if m is None:
        return "Error: Could not create map."
    
    return m._repr_html_()

@app.route('/calculate_routes', methods=['POST'])
def calculate_routes():
    """Quickly compare feasible charging stops and route alternatives."""
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Request body must be a JSON object'}), 400

        try:
            start_lat = float(data['start_lat'])
            start_lon = float(data['start_lon'])
            final_destination_lat = float(data['destination_lat'])
            final_destination_lon = float(data['destination_lon'])
            battery_percent = float(data.get('battery_percent', 70))
            reserve_percent = float(data.get('reserve_percent', 15))
            max_range_km = float(data.get('max_range_km', 250))
        except (KeyError, TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Start and destination coordinates, battery, reserve, and range must be valid numbers'}), 400
        run_aco_pso = data.get('run_aco_pso', True)
        if not isinstance(run_aco_pso, bool):
            return jsonify({'success': False, 'error': 'ACO/PSO comparison selection must be true or false'}), 400

        if not all(np.isfinite(value) for value in (start_lat, start_lon, final_destination_lat, final_destination_lon)):
            return jsonify({'success': False, 'error': 'Coordinates, battery, reserve, and range must be finite numbers'}), 400
        if not -90 <= start_lat <= 90 or not -90 <= final_destination_lat <= 90:
            return jsonify({'success': False, 'error': 'Start and destination latitudes must be between -90 and 90'}), 400
        if not np.isfinite(start_lon) or not -180 <= start_lon <= 180 or not np.isfinite(final_destination_lon) or not -180 <= final_destination_lon <= 180:
            return jsonify({'success': False, 'error': 'Start and destination longitudes must be between -180 and 180'}), 400
        if abs(start_lat - final_destination_lat) < 1e-8 and abs(start_lon - final_destination_lon) < 1e-8:
            return jsonify({'success': False, 'error': 'Choose a final destination different from the start point'}), 400
        if not 1 <= battery_percent <= 100:
            return jsonify({'success': False, 'error': 'Current charge must be between 1% and 100%'}), 400
        if not 0 <= reserve_percent < battery_percent:
            return jsonify({'success': False, 'error': 'Reserve must be at least 0% and lower than current charge'}), 400
        if not 20 <= max_range_km <= 1000:
            return jsonify({'success': False, 'error': 'Estimated full range must be between 20 and 1000 km'}), 400

        preference = data.get('route_preference', 'balanced')
        if not isinstance(preference, str) or preference not in ROUTE_PREFERENCES:
            return jsonify({'success': False, 'error': 'Choose a supported ride preference'}), 400
        preference_config = ROUTE_PREFERENCES[preference]
        w_distance, w_time, w_occupancy = preference_config['weights']
        usable_battery_level = (battery_percent - reserve_percent) / 100

        # Find the nearest road node to the exact map selection.
        start_node = find_nearest_node(G, start_lat, start_lon)
        if start_node is None:
            return jsonify({'success': False, 'error': 'Could not find nearest network node to start point'}), 400
        final_destination_node = find_nearest_node(G, final_destination_lat, final_destination_lon)
        if final_destination_node is None:
            return jsonify({'success': False, 'error': 'Could not find nearest network node to final destination'}), 400

        current_hour = datetime.now().hour

        traffic_lookup = {}
        for row in traffic_df.itertuples(index=False):
            traffic_lookup[(row.edge_u, row.edge_v, row.hour)] = {
                'travel_time': row.travel_time_seconds,
                'congestion': row.congestion_level,
                'speed': row.actual_speed_kmh
            }

        def edge_data_for_path(from_node, to_node):
            edge_options = G.get_edge_data(from_node, to_node)
            return next(iter(edge_options.values())) if edge_options else None

        def travel_time_for_edge(from_node, to_node, edge_data):
            traffic = traffic_lookup.get((from_node, to_node, current_hour))
            if traffic:
                return float(traffic['travel_time']), traffic
            speed = float(edge_data.get('speed_limit', 40) or 40)
            return (float(edge_data.get('length', 100)) / 1000) / speed * 3600, {}

        def battery_consumption(distance_km, speed_kmh, congestion, battery_level):
            efficiency = 1.3 if battery_level < 0.1 else 1.15 if battery_level < 0.2 else 1.0
            speed_factor = 1.2 if speed_kmh is not None and speed_kmh > 80 else (
                1.1 if speed_kmh is not None and speed_kmh > 60 else (
                    1.15 if speed_kmh is not None and speed_kmh < 20 else 1.0
                )
            )
            congestion_factor = 1.25 if congestion is not None and congestion > 0.7 else (
                1.1 if congestion is not None and congestion > 0.4 else 1.0
            )
            return min(distance_km / max_range_km * efficiency * speed_factor * congestion_factor, 1.0)

        def station_occupancy(station):
            occupancy = float(station.get('base_occupancy', 0.5))
            return min(occupancy * 1.3, 0.95) if current_hour in [8, 9, 17, 18, 19] else occupancy

        # One forward and one reverse Dijkstra run replace repeated searches per station.
        start_distances, start_paths = nx.single_source_dijkstra(G, start_node, weight='length')
        reverse_graph = G.reverse(copy=False)
        _, reverse_destination_paths = nx.single_source_dijkstra(
            reverse_graph, final_destination_node, weight='length'
        )

        candidates = []
        station_rows = list(stations_df.iterrows())

        def measure_leg(path, initial_battery):
            distance_m = 0.0
            travel_time_seconds = 0.0
            battery_remaining = initial_battery
            for path_index in range(len(path) - 1):
                from_node = path[path_index]
                to_node = path[path_index + 1]
                edge_data = edge_data_for_path(from_node, to_node)
                if edge_data is None:
                    return None
                edge_distance_m = float(edge_data.get('length', 0))
                travel_time, traffic = travel_time_for_edge(from_node, to_node, edge_data)
                if not np.isfinite(travel_time):
                    return None
                battery_used = battery_consumption(
                    edge_distance_m / 1000,
                    traffic.get('speed'),
                    traffic.get('congestion'),
                    battery_remaining
                )
                safety_margin = 0.05 if battery_remaining < 0.2 else 0.0
                if battery_remaining < battery_used + safety_margin:
                    return None
                battery_remaining -= battery_used
                distance_m += edge_distance_m
                travel_time_seconds += travel_time
            return {
                'distance_m': distance_m,
                'travel_time_seconds': travel_time_seconds,
                'remaining_battery': battery_remaining
            }

        for station_index, station in station_rows:
            station_node = station['node_id']
            try:
                path_to_station = start_paths.get(station_node)
                reverse_path_to_station = reverse_destination_paths.get(station_node)
                if not path_to_station or not reverse_path_to_station:
                    continue

                path_to_destination = list(reversed(reverse_path_to_station))

                first_leg = measure_leg(path_to_station, usable_battery_level)
                post_charge_battery = max(0.0, 1.0 - reserve_percent / 100)
                second_leg = measure_leg(path_to_destination, post_charge_battery)
                if first_leg is None or second_leg is None:
                    continue
                if first_leg['distance_m'] == 0:
                    continue

                occupancy = station_occupancy(station)
                waiting_seconds = occupancy * 900
                total_distance_m = first_leg['distance_m'] + second_leg['distance_m']
                total_time_seconds = first_leg['travel_time_seconds'] + second_leg['travel_time_seconds'] + waiting_seconds
                distance_score = w_distance * (total_distance_m / 1000)
                time_score = w_time * (total_time_seconds / 60)
                occupancy_score = w_occupancy * (waiting_seconds / 60)
                score = distance_score + time_score + occupancy_score

                candidates.append({
                    'station_index': station_index,
                    'node_id': station_node,
                    'name': str(station['name']),
                    'latitude': float(station['latitude']),
                    'longitude': float(station['longitude']),
                    'path': path_to_station,
                    'path_to_destination': path_to_destination,
                    'score': float(score),
                    'distance_km': total_distance_m / 1000,
                    'driving_minutes': (first_leg['travel_time_seconds'] + second_leg['travel_time_seconds']) / 60,
                    'wait_minutes': waiting_seconds / 60,
                    'first_leg_distance_km': first_leg['distance_m'] / 1000,
                    'second_leg_distance_km': second_leg['distance_m'] / 1000,
                    'distance_score': distance_score,
                    'time_score': time_score,
                    'occupancy_score': occupancy_score,
                    'station_arrival_battery': first_leg['remaining_battery'],
                    'final_arrival_battery': second_leg['remaining_battery']
                })
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, TypeError, ValueError):
                continue

        if not candidates:
            return jsonify({
                'success': False,
                'error': 'No charging station is reachable without using the reserved battery. Increase current charge, reduce reserve, or check the vehicle range.'
            }), 400

        candidates.sort(key=lambda candidate: candidate['score'])
        best_candidate = candidates[0]
        best_station_node = best_candidate['node_id']
        best_distance = best_candidate['distance_km']
        station_info = stations_df.loc[best_candidate['station_index']]
        best_station_name = str(station_info['name'])
        best_station_lat = float(station_info['latitude'])
        best_station_lon = float(station_info['longitude'])
        best_station_node = int(best_station_node)  # Convert int64 to int
        distance_path = best_candidate['path']

        def preference_edge_cost(from_node, to_node, edge_options):
            if G.is_multigraph():
                options = edge_options.values()
            else:
                options = (edge_options,)
            costs = []
            for edge_data in options:
                length_km = float(edge_data.get('length', 0)) / 1000
                travel_seconds, _ = travel_time_for_edge(from_node, to_node, edge_data)
                costs.append(w_distance * length_km + w_time * travel_seconds / 60)
            return min(costs) if costs else float('inf')

        preference_path = nx.shortest_path(
            G, start_node, best_station_node, weight=preference_edge_cost
        )
        if measure_leg(preference_path, usable_battery_level) is None:
            preference_path = distance_path

        def build_route(path, route_name, color):
            first_leg = measure_leg(path, usable_battery_level)
            second_leg = measure_leg(
                best_candidate['path_to_destination'],
                max(0.0, 1.0 - reserve_percent / 100)
            )
            if first_leg is None or second_leg is None:
                return None
            total_wait_seconds = station_occupancy(station_info) * 900
            route_cost = (
                w_distance * ((first_leg['distance_m'] + second_leg['distance_m']) / 1000) +
                w_time * ((first_leg['travel_time_seconds'] + second_leg['travel_time_seconds'] + total_wait_seconds) / 60) +
                w_occupancy * (total_wait_seconds / 60)
            )
            coords = [[start_lat, start_lon]]
            for node in path:
                node_data = G.nodes[node]
                node_coords = [float(node_data['y']), float(node_data['x'])]
                if node_coords != coords[-1]:
                    coords.append(node_coords)
            station_coords = [best_station_lat, best_station_lon]
            if station_coords != coords[-1]:
                coords.append(station_coords)
            for node in best_candidate['path_to_destination'][1:]:
                node_data = G.nodes[node]
                node_coords = [float(node_data['y']), float(node_data['x'])]
                if node_coords != coords[-1]:
                    coords.append(node_coords)
            final_coords = [final_destination_lat, final_destination_lon]
            if final_coords != coords[-1]:
                coords.append(final_coords)
            return {
                'name': route_name,
                'coords': coords,
                'color': color,
                'weight': 6,
                'cost': float(route_cost)
            }

        routes_data = [
            build_route(distance_path, 'Shortest-distance route', '#dc2626'),
            build_route(preference_path, 'Ride-preference route', '#2563eb')
        ]
        routes_data = [route for route in routes_data if route is not None]

        optimizer_results = {
            'ACO': {'message': 'Not run (disabled for speed).'},
            'PSO': {'message': 'Not run (disabled for speed).'}
        }
        optimizer_explanation = (
            'These are real ACO (8 ants × 8 iterations) and PSO (8 particles × 5 iterations) runs on the recommended charger. '
            'They may not find a valid battery-feasible path within the limited search.'
            if run_aco_pso else
            'ACO and PSO were not run for this request. Enable “Include ACO & PSO comparison” to run both with bounded iterations.'
        )

        if run_aco_pso:
            selected_station_df = stations_df.loc[[best_candidate['station_index']]].copy()
            optimizer_configs = (
                ('ACO', 'aco_algorithm', 'ACO_EV_Router', {'n_ants': 8, 'n_iterations': 8}, '#ea580c'),
                ('PSO', 'pso_algorithm', 'PSO_EV_Router', {'n_particles': 8, 'n_iterations': 5}, '#7c3aed')
            )
            for optimizer_name, module_name, class_name, parameters, color in optimizer_configs:
                optimizer_started = time.perf_counter()
                try:
                    optimizer_module = __import__(module_name, fromlist=[class_name])
                    optimizer_class = getattr(optimizer_module, class_name)
                    optimizer = optimizer_class(
                        graph=G,
                        stations_df=selected_station_df,
                        traffic_df=traffic_df,
                        start_node=start_node,
                        current_hour=current_hour,
                        battery_level=min(usable_battery_level, 0.999999),
                        max_range_km=max_range_km,
                        w_distance=w_distance,
                        w_time=w_time,
                        w_occupancy=w_occupancy,
                        **parameters
                    )
                    optimizer_path, _, _ = optimizer.run(verbose=False)
                    elapsed_seconds = time.perf_counter() - optimizer_started
                    if optimizer_path and optimizer_path[0] == start_node and optimizer_path[-1] == best_station_node:
                        optimizer_route = build_route(
                            optimizer_path,
                            optimizer_name + ' optimized route',
                            color
                        )
                        if optimizer_route:
                            optimizer_route['dash_array'] = '8 7'
                            routes_data.append(optimizer_route)
                            optimizer_results[optimizer_name] = {
                                'message': 'Route found · ' + str(len(optimizer_path)) + ' road nodes · ' + format(elapsed_seconds, '.2f') + ' s'
                            }
                        else:
                            optimizer_results[optimizer_name] = {
                                'message': 'Candidate route did not satisfy the battery reserve constraint · ' + format(elapsed_seconds, '.2f') + ' s'
                            }
                    else:
                        optimizer_results[optimizer_name] = {
                            'message': 'No valid route found within the bounded search · ' + format(elapsed_seconds, '.2f') + ' s'
                        }
                except Exception as optimizer_error:
                    import traceback
                    traceback.print_exc()
                    optimizer_results[optimizer_name] = {
                        'message': 'Optimizer error: ' + str(optimizer_error)
                    }

        score_rank_by_candidate = {
            id(candidate): rank
            for rank, candidate in enumerate(candidates, start=1)
        }

        def distance_from_selected_station(candidate):
            lat1 = np.radians(best_station_lat)
            lon1 = np.radians(best_station_lon)
            lat2 = np.radians(candidate['latitude'])
            lon2 = np.radians(candidate['longitude'])
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            haversine = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
            return 6371 * 2 * np.arcsin(np.sqrt(min(1.0, haversine)))

        nearby_candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate is not best_candidate,
                distance_from_selected_station(candidate),
                candidate['score']
            )
        )
        next_candidate = candidates[1] if len(candidates) > 1 else None
        comparison_candidates = [best_candidate]
        comparison_candidate_ids = {id(best_candidate)}
        if next_candidate:
            comparison_candidates.append(next_candidate)
            comparison_candidate_ids.add(id(next_candidate))
        for candidate in nearby_candidates:
            if id(candidate) not in comparison_candidate_ids:
                comparison_candidates.append(candidate)
                comparison_candidate_ids.add(id(candidate))
        comparison_candidates = comparison_candidates[:5]
        if next_candidate:
            score_difference = next_candidate['score'] - best_candidate['score']
            station_reason = (
                best_station_name + ' was selected because it has the lowest weighted score (' +
                format(best_candidate['score'], '.2f') + ') among stations reachable while preserving your reserve. ' +
                'The next-best feasible option, ' + next_candidate['name'] + ', scores ' +
                format(next_candidate['score'], '.2f') + ' (' + format(score_difference, '.2f') +
                ' higher), based on total trip distance, estimated driving time, and station wait.'
            )
        else:
            station_reason = (
                best_station_name + ' was selected because it is the only station that passes the battery-reserve '
                'checks for both trip legs.'
            )

        # Convert all numpy/pandas types to native Python types for JSON serialization
        result_data = {
            'success': True,
            'routes': routes_data,
            'start_lat': float(data['start_lat']),  # User's selected start point
            'start_lon': float(data['start_lon']),
            'destination_lat': float(best_station_lat) if best_station_lat is not None else None,  # Kept as station aliases for existing clients.
            'destination_lon': float(best_station_lon) if best_station_lon is not None else None,
            'destination_name': str(best_station_name),
            'station_lat': float(best_station_lat),
            'station_lon': float(best_station_lon),
            'station_name': str(best_station_name),
            'final_destination_lat': final_destination_lat,
            'final_destination_lon': final_destination_lon,
            'final_destination_name': 'Selected map location',
            'destination_node': int(best_station_node),
            'selected_station_node': int(best_station_node),
            'route_preference': preference,
            'route_preference_label': preference_config['label'],
            'battery_percent': battery_percent,
            'reserve_percent': reserve_percent,
            'usable_battery_percent': usable_battery_level * 100,
            'station_arrival_percent': min(100.0, reserve_percent + best_candidate['station_arrival_battery'] * 100),
            'final_arrival_percent': min(100.0, reserve_percent + best_candidate['final_arrival_battery'] * 100),
            'estimated_remaining_percent': min(100.0, reserve_percent + best_candidate['final_arrival_battery'] * 100),
            'distance_km': float(best_distance),
            'feasible_station_count': len(candidates),
            'evaluated_station_count': len(station_rows),
            'station_reason': station_reason,
            'station_score_weights': {
                'distance': w_distance,
                'time': w_time,
                'occupancy': w_occupancy
            },
            'station_comparison': [
                {
                    'score_rank': score_rank_by_candidate[id(candidate)],
                    'name': candidate['name'],
                    'selected': candidate is best_candidate,
                    'distance_from_selected_km': distance_from_selected_station(candidate),
                    'distance_km': candidate['distance_km'],
                    'driving_minutes': candidate['driving_minutes'],
                    'wait_minutes': candidate['wait_minutes'],
                    'score': candidate['score'],
                    'distance_score': candidate['distance_score'],
                    'time_score': candidate['time_score'],
                    'occupancy_score': candidate['occupancy_score'],
                    'station_arrival_percent': min(100.0, reserve_percent + candidate['station_arrival_battery'] * 100),
                    'final_arrival_percent': min(100.0, reserve_percent + candidate['final_arrival_battery'] * 100)
                }
                for candidate in comparison_candidates
            ],
            'optimizer_results': optimizer_results,
            'optimizer_explanation': optimizer_explanation
        }
        
        return jsonify(result_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("EV CHARGING ROUTE OPTIMIZATION - WEB SERVER")
    print("="*60)
    
    if not load_data():
        print("\nERROR: Failed to load data. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Server starting...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
