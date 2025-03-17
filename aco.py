import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
import random
import numpy as np

# Configure OSMnx caching and logging
ox.settings.use_cache = True
ox.settings.log_console = True

###############################################################################
# 1) LOAD / CACHING THE ROAD NETWORK
###############################################################################
def load_road_network(place="Coimbatore, India", cache_dir="road_cache"):
    """
    Loads (or downloads) a road network for the specified place.
    Saves/loads from a local GraphML cache to avoid re-downloading.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_place = place.replace(" ", "_").replace(",", "")
    cache_file = os.path.join(cache_dir, f"{safe_place}_network.graphml")
    
    # Example fallback to a major roads .graphml if you have it locally:
    major_road_network = r"C:\Users\DELL\Documents\Amrita\4th year\ArcGis\major_road_cache\Coimbatore_India_major.graphml"

    # Adjust logic depending on your preference:
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

###############################################################################
# 2) FETCH POLICE STATIONS & NEAREST ROAD NETWORK NODES
###############################################################################
def get_police_station_nodes(G, place="Coimbatore, India"):
    """
    1) Fetch police stations from OSM for the given place using amenities tag.
    2) For each police station geometry, find the nearest node in the road network G.
    3) Return a list of valid node IDs corresponding to these stations.
       If none found, returns an empty list.
    """
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

###############################################################################
# 3) ACO CONFIG AND OPTIMIZER
###############################################################################
class ACOConfig:
    def __init__(self):
        self.num_vehicles = 5
        self.num_iterations = 50
        # ACO parameters
        self.alpha = 1.0
        self.beta = 2.0
        self.evaporation_rate = 0.1
        self.Q = 100
        
        # Constraints/penalties
        self.max_route_distance = 15000
        self.penalty_factor = 10000   # Overlap penalty for edges used >3 times
        # NEW: coverage penalty factor – punishes uncovered edges
        self.coverage_penalty_factor = 8000

class RouteOptimizerACO:
    def __init__(self, G, config, station_nodes=None):
        self.G = G
        self.config = config
        self.station_nodes = station_nodes if station_nodes is not None else []

        # Build edge caches
        self.edge_length_cache = {}
        self.adjacency_cache = {}
        self._build_edge_length_cache()
        self._build_adjacency_cache()

        # Keep track of all directed edges (u->v)
        self.all_edges = set(self.edge_length_cache.keys())

        # Initialize pheromone
        self.pheromone = {}
        init_pheromone = 1.0
        for (u, v) in self.edge_length_cache.keys():
            self.pheromone[(u, v)] = init_pheromone

        self.best_solution = None
        self.best_fitness = -float('inf')

    def _build_edge_length_cache(self):
        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 0)
            self.edge_length_cache[(u, v)] = length

    def _build_adjacency_cache(self):
        for node in self.G.nodes():
            self.adjacency_cache[node] = list(self.G.successors(node))

    def _probability_distribution(self, current_node):
        neighbors = self.adjacency_cache[current_node]
        if not neighbors:
            return []

        desirabilities = []
        for nbr in neighbors:
            edge_len = self.edge_length_cache.get((current_node, nbr), 1)
            if edge_len <= 0:
                edge_len = 0.001
            tau = self.pheromone[(current_node, nbr)] ** self.config.alpha
            eta = (1.0 / edge_len) ** self.config.beta
            desirabilities.append((nbr, tau * eta))

        total = sum(d for (_, d) in desirabilities)
        if total == 0:
            count = len(desirabilities)
            return [(n, 1.0 / count) for (n, _) in desirabilities]
        return [(n, d / total) for (n, d) in desirabilities]

    def _select_next_node(self, prob_dist):
        if not prob_dist:
            return None
        r = random.random()
        cumulative = 0.0
        for node, p in prob_dist:
            cumulative += p
            if r <= cumulative:
                return node
        return prob_dist[-1][0]

    def build_route(self):
        # Start from police station if available, else random
        if self.station_nodes:
            current_node = random.choice(self.station_nodes)
        else:
            current_node = random.choice(list(self.G.nodes()))

        route = [current_node]
        total_dist = 0

        while True:
            prob_dist = self._probability_distribution(current_node)
            next_node = self._select_next_node(prob_dist)
            if next_node is None:
                break

            dist = self.edge_length_cache.get((current_node, next_node), 0)
            if total_dist + dist > self.config.max_route_distance:
                break

            route.append(next_node)
            total_dist += dist
            current_node = next_node

        return route

    def build_solution(self):
        return [self.build_route() for _ in range(self.config.num_vehicles)]

    def fitness(self, solution):
        """
        A coverage-aware fitness function:
        1) Sum total route length.
        2) Subtract overlap penalty for edges used >3 times.
        3) Subtract coverage penalty for edges left uncovered.
        """
        total_length = 0
        edge_usage = {}
        covered_edges = set()

        for route in solution:
            route_length = 0
            for u, v in zip(route[:-1], route[1:]):
                length = self.edge_length_cache.get((u, v), 0)
                route_length += length
                covered_edges.add((u, v))
                edge_usage[(u, v)] = edge_usage.get((u, v), 0) + 1
            total_length += route_length

        # Overlap penalty for edges used by more than 3 vehicles
        overlap_penalty = 0
        for (u, v), count in edge_usage.items():
            if count > 3:
                length = self.edge_length_cache.get((u, v), 0)
                overlap_penalty += self.config.penalty_factor * (count - 3) * length

        # Coverage penalty for uncovered edges
        uncovered_edges = self.all_edges - covered_edges
        uncovered_penalty = 0
        for (u, v) in uncovered_edges:
            edge_len = self.edge_length_cache.get((u, v), 0)
            uncovered_penalty += self.config.coverage_penalty_factor * edge_len

        # Final fitness
        return total_length - overlap_penalty - uncovered_penalty

    def evaporate_pheromones(self):
        rho = self.config.evaporation_rate
        for e in self.pheromone:
            self.pheromone[e] = (1.0 - rho) * self.pheromone[e]

    def deposit_pheromones(self, solution, fit_value):
        deposit_amount = max(fit_value, 1)
        for route in solution:
            for u, v in zip(route[:-1], route[1:]):
                self.pheromone[(u, v)] += (self.config.Q * deposit_amount / 1000.0)

    def run(self):
        for iteration in range(self.config.num_iterations):
            solution = self.build_solution()
            fit_value = self.fitness(solution)
            if fit_value > self.best_fitness:
                self.best_fitness = fit_value
                self.best_solution = solution

            self.evaporate_pheromones()
            self.deposit_pheromones(solution, fit_value)
        return self.best_solution, self.best_fitness

###############################################################################
# 4) CONVERT SOLUTION TO GEOJSON FOR VISUALIZATION
###############################################################################
def solution_to_geojson(G, solution):
    """
    Convert a solution (list of routes) into a GeoJSON FeatureCollection.
    Each route is assigned a unique color for clarity.
    """
    # You can define more colors if needed
    colors = [
        "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
    ]
    features = []
    for vid, route in enumerate(solution):
        if len(route) < 2:
            continue
        lines = []
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                geom = edge_data[0].get('geometry', LineString([
                    (G.nodes[u]['x'], G.nodes[u]['y']),
                    (G.nodes[v]['x'], G.nodes[v]['y'])
                ]))
                lines.append(geom)
        if lines:
            color = colors[vid % len(colors)]
            features.append({
                "type": "Feature",
                "properties": {
                    "vehicle": vid + 1,
                    "stroke": color,
                    "stroke-width": 3,
                    "stroke-opacity": 0.8
                },
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [list(line.coords) for line in lines]
                }
            })
    return {"type": "FeatureCollection", "features": features}

def route_to_geojson(G, route, vehicle_id):
    """
    Convert a single vehicle route into a GeoJSON FeatureCollection
    for highlighting.
    """
    lines = []
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            geom = edge_data[0].get('geometry', LineString([
                (G.nodes[u]['x'], G.nodes[u]['y']),
                (G.nodes[v]['x'], G.nodes[v]['y'])
            ]))
            lines.append(geom)
    if lines:
        feature = {
            "type": "Feature",
            "properties": {
                "vehicle": vehicle_id,
                "stroke": "#FF0000",
                "stroke-width": 5,
                "stroke-opacity": 1.0
            },
            "geometry": {
                "type": "MultiLineString",
                "coordinates": [list(line.coords) for line in lines]
            }
        }
        return {"type": "FeatureCollection", "features": [feature]}
    else:
        return None

###############################################################################
# 5) STREAMLIT MAIN APP
###############################################################################
def main():
    st.set_page_config(layout="wide")
    st.title("🚔 City-Wide Vehicle Coverage (ACO w/ Coverage Objective)")

    # Sidebar controls for algorithm parameters
    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles (Ants)", 3, 50, 5)
        num_iterations = st.slider("Number of Iterations", 10, 500, 50)
        alpha = st.slider("Alpha (pheromone influence)", 0.0, 5.0, 1.0)
        beta = st.slider("Beta (distance influence)", 0.0, 5.0, 2.0)
        evaporation_rate = st.slider("Evaporation Rate (ρ)", 0.0, 1.0, 0.1)
        Q = st.slider("Pheromone Deposit (Q)", 1, 1000, 100)

        # Coverage penalty factor
        coverage_penalty = st.slider(
            "Coverage Penalty Factor", 
            1000, 20000, 8000, step=1000,
            help="Higher = stronger penalty for leaving edges uncovered."
        )

        run_btn = st.button("Run ACO Optimization")

        # If a solution exists, we can highlight a route
        if "aco_solution" in st.session_state:
            route_options = ["None"] + [f"Route {i+1}" for i in range(len(st.session_state['aco_solution']))]
            selected_route = st.selectbox("Select a route to highlight", options=route_options)
        else:
            selected_route = "None"

    # Run the algorithm only when the Run button is clicked
    if run_btn:
        config = ACOConfig()
        config.num_vehicles = num_vehicles
        config.num_iterations = num_iterations
        config.alpha = alpha
        config.beta = beta
        config.evaporation_rate = evaporation_rate
        config.Q = Q
        config.coverage_penalty_factor = coverage_penalty

        with st.spinner("Loading data and running ACO optimization..."):
            # 1) Load network
            G = load_road_network("Coimbatore, India")
            # 2) Get police stations
            station_nodes = get_police_station_nodes(G, "Coimbatore, India")
            st.write(f"Found {len(station_nodes)} police station node(s).")

            # 3) Run ACO
            optimizer = RouteOptimizerACO(G, config, station_nodes)
            best_solution, best_fit = optimizer.run()

            geojson_data = solution_to_geojson(G, best_solution)

            # Store results in session state
            st.session_state['aco_solution'] = best_solution
            st.session_state['aco_fit'] = best_fit
            st.session_state['graph'] = G
            st.session_state['geojson_data'] = geojson_data

    # If a solution exists, render the map
    if "aco_solution" in st.session_state:
        G = st.session_state['graph']
        best_solution = st.session_state['aco_solution']
        best_fit = st.session_state['aco_fit']
        geojson_data = st.session_state['geojson_data']

        # Fetch boundary polygon
        boundary_gdf = ox.geocode_to_gdf("Coimbatore, India")

        m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
        m.add_gdf(boundary_gdf, layer_name="Coimbatore Boundary",
                  style={"color": "#FF0000", "fillOpacity": 0.1})
        m.add_geojson(geojson_data, layer_name="Optimized Routes")

        # If a route is selected, highlight it
        if selected_route != "None":
            try:
                vehicle_idx = int(selected_route.split(" ")[1]) - 1
                if 0 <= vehicle_idx < len(best_solution):
                    highlight_geojson = route_to_geojson(G, best_solution[vehicle_idx], vehicle_idx + 1)
                    if highlight_geojson:
                        m.add_geojson(highlight_geojson, layer_name=f"Vehicle {vehicle_idx + 1} Highlight")
                        # Zoom to route
                        coords = []
                        for node in best_solution[vehicle_idx]:
                            lat = G.nodes[node]['y']
                            lon = G.nodes[node]['x']
                            coords.append([lat, lon])
                        if coords:
                            lats = [pt[0] for pt in coords]
                            lons = [pt[1] for pt in coords]
                            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
                            m.fit_bounds(bounds)
                else:
                    st.error("Invalid vehicle route number.")
            except Exception as e:
                st.error(f"Error processing route selection: {e}")

        st.success(f"Best Fitness: {best_fit:.1f}")
        m.to_streamlit()

if __name__ == "__main__":
    main()
