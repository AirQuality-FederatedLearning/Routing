import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import random
import numpy as np
import time
from shapely.geometry import Point, LineString
from tqdm import tqdm

ox.settings.use_cache = True
ox.settings.log_console = True

# Load and cache the road network of Coimbatore
def load_road_network(place="Coimbatore, India", cache_dir="road_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    safe_place = place.replace(" ", "_").replace(",", "")
    cache_file = os.path.join(cache_dir, f"{safe_place}_network.graphml")

    major_road_network = r"C:\Users\DELL\Documents\Amrita\4th year\ArcGis\major_road_cache\Coimbatore_India_major.graphml"

    if os.path.exists(major_road_network):
        st.write(f"Loading cached road network from {major_road_network}")
        G = ox.load_graphml(major_road_network)
    elif os.path.exists(cache_file):
        st.write(f"Loading cached road network from {cache_file}")
        G = ox.load_graphml(cache_file)
    else:
        st.write(f"Downloading road network for {place}")
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        ox.save_graphml(G, cache_file)

    return G

# Get police stations as starting points
def get_police_station_nodes(G, place="Coimbatore, India"):
    try:
        gdf_police = ox.features_from_place(place, tags={"amenity": "police"})
    except Exception as e:
        st.warning(f"Could not fetch police stations: {e}")
        return []

    if gdf_police.empty:
        st.info("No police stations found with the given tag. Falling back to random start nodes.")
        return []

    station_nodes = []
    for idx, row in gdf_police.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        point = geom.centroid if geom.geom_type != 'Point' else geom
        node_id = ox.distance.nearest_nodes(G, point.x, point.y)
        station_nodes.append(node_id)
    return list(set(station_nodes))

# Simulated Annealing Optimization
def simulated_annealing(G, station_nodes, max_iterations=1000, initial_temp=1000, cooling_rate=0.995):
    current_solution = random.sample(station_nodes, len(station_nodes))  # Random initial solution
    current_cost = calculate_total_distance(G, current_solution)
    best_solution = list(current_solution)
    best_cost = current_cost
    temperature = initial_temp

    for iteration in tqdm(range(max_iterations), desc="Simulated Annealing"):
        # Generate new solution by swapping two random nodes
        new_solution = current_solution[:]
        i, j = random.sample(range(len(station_nodes)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        new_cost = calculate_total_distance(G, new_solution)

        # Acceptance probability
        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
            current_solution = list(new_solution)
            current_cost = new_cost

        # Update best solution
        if new_cost < best_cost:
            best_solution = list(new_solution)
            best_cost = new_cost

        # Cooling
        temperature *= cooling_rate

    return best_solution, best_cost

# Calculate the total distance of a route
def calculate_total_distance(G, solution):
    total_distance = 0
    for i in range(len(solution) - 1):
        u, v = solution[i], solution[i+1]
        total_distance += nx.shortest_path_length(G, u, v, weight='length')
    return total_distance

# Convert route to GeoJSON for visualization
def solution_to_geojson(G, solution):
    features = []
    for i in range(len(solution) - 1):
        u, v = solution[i], solution[i+1]
        edge_data = G.get_edge_data(u, v)
        geom = edge_data[0]['geometry'] if edge_data else LineString([
            (G.nodes[u]['x'], G.nodes[u]['y']),
            (G.nodes[v]['x'], G.nodes[v]['y'])
        ])
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": list(geom.coords)
            },
            "properties": {
                "vehicle": 1,  # Single vehicle for simplicity
                "stroke": "#0000FF",
                "stroke-width": 3,
                "stroke-opacity": 0.8
            }
        })
    return {"type": "FeatureCollection", "features": features}

# Streamlit interface
def main():
    st.set_page_config(layout="wide")
    st.title("Simulated Annealing for Police Patrol Route Optimization")

    with st.sidebar:
        num_iterations = st.slider("Number of Iterations", 100, 2000, 1000)
        initial_temp = st.slider("Initial Temperature", 100, 10000, 1000)
        cooling_rate = st.slider("Cooling Rate", 0.9, 1.0, 0.995)
        run_btn = st.button("Run Simulated Annealing")

    if run_btn:
        start_time = time.time()
        with st.spinner("Loading road network..."):
            G = load_road_network("Coimbatore, India")

        with st.spinner("Fetching police stations..."):
            station_nodes = get_police_station_nodes(G, "Coimbatore, India")

        with st.spinner("Running Simulated Annealing..."):
            best_solution, best_cost = simulated_annealing(G, station_nodes, num_iterations, initial_temp, cooling_rate)

        geojson_data = solution_to_geojson(G, best_solution)
        m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
        m.add_geojson(geojson_data, layer_name="Optimized Patrol Routes")

        st.success(f"Best Solution Found: Total Distance = {best_cost:.2f} meters")
        m.to_streamlit()

        end_time = time.time()
        st.write(f"Execution Time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
