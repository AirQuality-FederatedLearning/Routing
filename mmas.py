import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
import random
import numpy as np
import time
from tqdm import trange

ox.settings.use_cache = True
ox.settings.log_console = True

###############################################################################
# 1) LOAD / CACHING THE ROAD NETWORK
###############################################################################
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

###############################################################################
# 2) FETCH POLICE STATIONS & NEAREST ROAD NETWORK NODES
###############################################################################
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

###############################################################################
# 3) MMAS CONFIG AND OPTIMIZER
###############################################################################
class MMASConfig:
    def __init__(self):
        self.num_vehicles = 5
        self.num_iterations = 50
        # Heuristic parameters
        self.alpha = 1.0
        self.beta = 2.0
        # Evaporation
        self.evaporation_rate = 0.1
        self.Q = 100
        # Overlap / coverage constraints
        self.max_route_distance = 15000
        self.penalty_factor = 10000
        self.coverage_penalty_factor = 8000
        # MMAS-specific bounds
        self.tau_min = 0.1
        self.tau_max = 5.0

class MMASOptimizer:
    def __init__(self, G, config, station_nodes=None):
        self.G = G
        self.config = config
        self.station_nodes = station_nodes if station_nodes else []

        self.edge_length_cache = {}
        self.adjacency_cache = {}
        self._build_edge_length_cache()
        self._build_adjacency_cache()

        self.all_edges = set(self.edge_length_cache.keys())

        # Initialize pheromone to some mid-range
        init_pher = 1.0
        self.pheromone = { (u,v): init_pher for (u,v) in self.edge_length_cache.keys() }

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
            length = self.edge_length_cache.get((current_node, nbr), 1)
            if length <= 0:
                length = 0.001
            tau = self.pheromone[(current_node, nbr)] ** self.config.alpha
            eta = (1.0 / length) ** self.config.beta
            desirabilities.append((nbr, tau * eta))
        total = sum(d for (_, d) in desirabilities)
        if total == 0:
            return [(n, 1.0/len(desirabilities)) for (n,_) in desirabilities]
        return [(n, d/total) for (n,d) in desirabilities]

    def _select_next_node(self, prob_dist):
        if not prob_dist:
            return None
        r = random.random()
        cum = 0.0
        for node, prob in prob_dist:
            cum += prob
            if r <= cum:
                return node
        return prob_dist[-1][0]

    def build_route(self):
        start_node = random.choice(self.station_nodes) if self.station_nodes else random.choice(list(self.G.nodes()))
        route = [start_node]
        total_dist = 0
        current = start_node
        while True:
            p_dist = self._probability_distribution(current)
            nxt = self._select_next_node(p_dist)
            if nxt is None:
                break
            dist = self.edge_length_cache.get((current, nxt), 0)
            if total_dist + dist > self.config.max_route_distance:
                break
            route.append(nxt)
            total_dist += dist
            current = nxt
        return route

    def build_solution(self):
        return [self.build_route() for _ in range(self.config.num_vehicles)]

    def fitness(self, solution):
        total_length = 0
        edge_usage = {}
        covered_edges = set()
        for route in solution:
            route_length = 0
            for u,v in zip(route[:-1], route[1:]):
                length = self.edge_length_cache.get((u,v), 0)
                route_length += length
                covered_edges.add((u,v))
                edge_usage[(u,v)] = edge_usage.get((u,v), 0) + 1
            total_length += route_length
        # Overlap penalty
        overlap_penalty = 0
        for (u,v), count in edge_usage.items():
            if count > 3:
                length = self.edge_length_cache.get((u,v), 0)
                overlap_penalty += self.config.penalty_factor * (count - 3) * length
        # Coverage penalty
        uncovered = self.all_edges - covered_edges
        coverage_penalty = 0
        for (u,v) in uncovered:
            length = self.edge_length_cache.get((u,v), 0)
            coverage_penalty += self.config.coverage_penalty_factor * length
        return total_length - overlap_penalty - coverage_penalty

    def evaporate_pheromones(self):
        rho = self.config.evaporation_rate
        for e in self.pheromone:
            self.pheromone[e] = (1-rho)*self.pheromone[e]

    def deposit_pheromones(self, best_sol, best_fit):
        # In MMAS, we deposit only from the best solution
        deposit = max(best_fit, 1)
        for route in best_sol:
            for u,v in zip(route[:-1], route[1:]):
                self.pheromone[(u,v)] += (self.config.Q * deposit / 1000.0)
        # enforce tau_min, tau_max
        for e in self.pheromone:
            if self.pheromone[e] < self.config.tau_min:
                self.pheromone[e] = self.config.tau_min
            elif self.pheromone[e] > self.config.tau_max:
                self.pheromone[e] = self.config.tau_max

    def run(self):
        for _ in trange(self.config.num_iterations, desc="MMAS Iteration"):
            solution = self.build_solution()
            fit_val = self.fitness(solution)
            # Update global best
            if fit_val > self.best_fitness:
                self.best_fitness = fit_val
                self.best_solution = solution
            self.evaporate_pheromones()
            # deposit from best known solution
            self.deposit_pheromones(self.best_solution, self.best_fitness)

        return self.best_solution, self.best_fitness

###############################################################################
# 4) CONVERT SOLUTION TO GEOJSON FOR VISUALIZATION
###############################################################################
def solution_to_geojson(G, solution):
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
# 5) STREAMLIT MAIN APP - MMAS
###############################################################################
def main():
    st.set_page_config(layout="wide")
    st.title("🚔 City-Wide Vehicle Coverage (MMAS Variant)")

    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles (Ants)", 3, 50, 5)
        num_iterations = st.slider("Number of Iterations", 10, 500, 50)
        alpha = st.slider("Alpha (pheromone influence)", 0.0, 5.0, 1.0)
        beta = st.slider("Beta (distance influence)", 0.0, 5.0, 2.0)
        evap_rate = st.slider("Evaporation Rate (ρ)", 0.0, 1.0, 0.1)
        Q = st.slider("Pheromone Deposit (Q)", 1, 1000, 100)

        tau_min = st.slider("τ_min", 0.01, 1.0, 0.1, step=0.01)
        tau_max = st.slider("τ_max", 1.0, 10.0, 5.0)

        coverage_penalty = st.slider("Coverage Penalty Factor", 1000, 20000, 8000, step=1000)
        run_btn = st.button("Run MMAS")

        if "mmas_solution" in st.session_state:
            route_opts = ["None"] + [f"Route {i+1}" for i in range(len(st.session_state['mmas_solution']))]
            selected_route = st.selectbox("Select a route to highlight", options=route_opts)
        else:
            selected_route = "None"

    if run_btn:
        start_time = time.time()
        config = MMASConfig()
        config.num_vehicles = num_vehicles
        config.num_iterations = num_iterations
        config.alpha = alpha
        config.beta = beta
        config.evaporation_rate = evap_rate
        config.Q = Q
        config.coverage_penalty_factor = coverage_penalty
        config.tau_min = tau_min
        config.tau_max = tau_max

        with st.spinner("Loading data and running MMAS..."):
            G = load_road_network("Coimbatore, India")
            station_nodes = get_police_station_nodes(G, "Coimbatore, India")

            optimizer = MMASOptimizer(G, config, station_nodes)
            best_solution, best_fit = optimizer.run()

            geojson_data = solution_to_geojson(G, best_solution)

            st.session_state['mmas_graph'] = G
            st.session_state['mmas_solution'] = best_solution
            st.session_state['mmas_fit'] = best_fit
            st.session_state['mmas_geojson'] = geojson_data

        end_time = time.time()
        st.write(f"Time taken: {end_time - start_time:.2f} seconds.")
    if "mmas_solution" in st.session_state:
        G = st.session_state['mmas_graph']
        best_solution = st.session_state['mmas_solution']
        best_fit = st.session_state['mmas_fit']
        geojson_data = st.session_state['mmas_geojson']

        boundary_gdf = ox.geocode_to_gdf("Coimbatore, India")
        m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
        m.add_gdf(boundary_gdf, layer_name="Coimbatore Boundary",
                  style={"color": "#FF0000", "fillOpacity": 0.1})
        m.add_geojson(geojson_data, layer_name="Optimized Routes (MMAS)")

        if selected_route != "None":
            try:
                idx = int(selected_route.split(" ")[1]) - 1
                if 0 <= idx < len(best_solution):
                    highlight_geojson = route_to_geojson(G, best_solution[idx], idx + 1)
                    if highlight_geojson:
                        m.add_geojson(highlight_geojson, layer_name=f"Vehicle {idx+1} Highlight")
                        coords = []
                        for node in best_solution[idx]:
                            lat = G.nodes[node]['y']
                            lon = G.nodes[node]['x']
                            coords.append([lat, lon])
                        if coords:
                            lats = [pt[0] for pt in coords]
                            lons = [pt[1] for pt in coords]
                            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
                            m.fit_bounds(bounds)
            except Exception as e:
                st.error(f"Error highlighting route: {e}")

        st.success(f"Best Fitness: {best_fit:.1f}")
        m.to_streamlit()

if __name__ == "__main__":
    main()
