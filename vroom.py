import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import numpy as np
import requests
from shapely.geometry import Point, LineString
import vroom

ox.settings.use_cache = True
ox.settings.log_console = True

# Configuration
PLACE = "Coimbatore, India"
NUM_VEHICLES = 5
MAX_ROUTE_DURATION = 3600  # 1 hour in seconds
MAX_ROUTE_DURATION_MS = MAX_ROUTE_DURATION * 1000  # Convert to milliseconds
OSRM_URL = "https://router.project-osrm.org/table/v1/driving/"

def get_road_network():
    """Get road network graph for the specified place."""
    return ox.graph_from_place(PLACE, network_type="drive", simplify=True)

def get_police_stations(place=PLACE):
    """Get police station locations as Shapely Points."""
    try:
        gdf = ox.features_from_place(place, tags={"amenity": "police"})
        return [Point(row.geometry.x, row.geometry.y) for _, row in gdf.iterrows()]
    except Exception as e:
        st.error(f"Error fetching police stations: {e}")
        return []

def create_time_matrix(locations):
    """Create time matrix using OSRM, returning durations in milliseconds as integers."""
    coordinates = ";".join([f"{lon},{lat}" for lon, lat in locations])
    url = f"{OSRM_URL}{coordinates}?sources=all&annotations=duration"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Convert seconds to milliseconds and cast to integers
        durations = (np.array(data['durations']) * 1000).astype(int)
        return durations
    except Exception as e:
        st.error(f"OSRM request failed: {e}")
        return None

def optimize_routes(time_matrix, num_vehicles):
    """Solve vehicle routing problem with VROOM."""
    if time_matrix is None:
        return None

    problem = vroom.InputProblem()

    # Add the time matrix (durations in milliseconds)
    problem.set_durations_matrix(time_matrix.tolist())

    # Add vehicles
    for i in range(num_vehicles):
        vehicle = vroom.Vehicle(
            id=i,
            start=i,
            end=i,
            time_window=vroom.TimeWindow(0, MAX_ROUTE_DURATION_MS),
        )
        problem.add_vehicle(vehicle)

    # Add jobs (patrol points)
    num_jobs = time_matrix.shape[0] - num_vehicles
    for j in range(num_jobs):
        job = vroom.Job(
            id=j,
            location=num_vehicles + j,
            service=0  # No service time
        )
        problem.add_job(job)

    # Solve the problem
    try:
        solution = vroom.solve(problem)
    except Exception as e:
        st.error(f"VROOM solving failed: {e}")
        return None

    # Extract routes
    routes = []
    for vehicle_route in solution.routes:
        route = []
        for step in vehicle_route.steps:
            route.append(step.location)
        routes.append(route)

    return routes

def routes_to_geojson(G, routes, locations):
    """Convert routes to GeoJSON for visualization."""
    features = []
    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
    
    for idx, route in enumerate(routes):
        route_points = []
        for node in route:
            lon, lat = locations[node]
            nearest = ox.nearest_nodes(G, X=[lon], Y=[lat])[0]
            route_points.append((G.nodes[nearest]['x'], G.nodes[nearest]['y']))
        
        if len(route_points) > 1:
            line = LineString(route_points)
            features.append({
                "type": "Feature",
                "properties": {
                    "vehicle": idx,
                    "color": colors[idx % len(colors)],
                    "stroke-width": 3
                },
                "geometry": line.__geo_interface__
            })
    
    return {"type": "FeatureCollection", "features": features}

def main():
    st.set_page_config(layout="wide")
    st.title("Police Patrol Route Optimization")
    
    # Load data
    with st.spinner("Loading road network..."):
        G = get_road_network()
        police_stations = get_police_stations()
        
        if not police_stations:
            st.error("No police stations found!")
            return
        
        if len(police_stations) < NUM_VEHICLES:
            st.error(f"Not enough police stations. Found {len(police_stations)}, required {NUM_VEHICLES}.")
            return
            
        # Get road network nodes as potential patrol locations
        nodes = list(G.nodes())
        sample_nodes = nodes[:100]  # Use first 100 nodes for demo
        
        # Convert to (lon, lat) format
        locations = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in sample_nodes]
        police_locations = [(pt.x, pt.y) for pt in police_stations][:NUM_VEHICLES]
        
        # Combine police stations and patrol locations
        all_locations = police_locations + locations
    
    # Create time matrix
    with st.spinner("Calculating travel times..."):
        time_matrix = create_time_matrix(all_locations)
        if time_matrix is None:
            return
    
    # Optimize routes
    with st.spinner("Optimizing patrol routes..."):
        routes = optimize_routes(time_matrix, NUM_VEHICLES)
        
        if not routes:
            st.error("No valid routes found!")
            return
    
    # Visualization
    geojson = routes_to_geojson(G, routes, all_locations)
    
    m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
    m.add_gdf(ox.geocode_to_gdf(PLACE), layer_name="City Boundary")
    m.add_geojson(geojson, layer_name="Patrol Routes")
    
    # Add police stations
    for idx, (lon, lat) in enumerate(police_locations):
        m.add_marker(location=[lat, lon], popup=f"Police Station {idx}")
    
    st.success("Optimization complete!")
    m.to_streamlit()

if __name__ == "__main__":
    main()