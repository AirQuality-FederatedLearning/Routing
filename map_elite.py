import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import random
import numpy as np
from shapely.geometry import LineString
import time

ox.settings.use_cache = True
ox.settings.log_console = True

###############################################################################
# 1) LOAD / CACHING THE ROAD NETWORK
###############################################################################
def load_road_network(place="Coimbatore, India", cache_dir="road_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    safe_place = place.replace(" ", "_").replace(",", "")
    cache_file = os.path.join(cache_dir, f"{safe_place}_network.graphml")

    major_road_network = r"C:\Users\DELL\Documents\Amrita\4th year\ArcGis\major_road_cache\\road_cache\Coimbatore_India_highways_primary.graphml"

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
        st.info("No police stations found. Falling back to random start nodes.")
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
# 3) MAP-Elites CONFIG
###############################################################################
class MAPElitesConfig:
    def __init__(self):
        self.num_vehicles = 5
        self.initial_population_size = 20
        self.num_iterations = 50
        self.mutation_rate = 0.2
        self.coverage_bins = [0, 5000, 10000, 15000, 20000, 25000]
        self.overlap_bins = [0, 1000, 2000, 3000, 4000, 5000]
        self.coverage_factor = 5000
        self.overlap_penalty_factor = 10000
        self.max_route_distance = 15000

###############################################################################
# 4) MAP-Elites OPTIMIZER
###############################################################################
class MAPElitesOptimizer:
    def __init__(self, G, config, start_nodes=None):
        self.G = G
        self.config = config
        self.start_nodes = start_nodes

        self.adjacency_cache = {}
        self.edge_length_cache = {}
        for node in self.G.nodes():
            self.adjacency_cache[node] = list(self.G.successors(node))
        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 0)
            self.edge_length_cache[(u, v)] = length
        self.all_edges = set(self.edge_length_cache.keys())
        self.archive = {}

    def build_random_route(self):
        nodes_list = list(self.G.nodes())
        if not nodes_list:
            return []
        current = random.choice(self.start_nodes) if self.start_nodes else random.choice(nodes_list)
        route = [current]
        dist_so_far = 0.0

        while True:
            neighbors = self.adjacency_cache[current]
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            edge_len = self.edge_length_cache.get((current, next_node), 0)
            if dist_so_far + edge_len > self.config.max_route_distance:
                break
            route.append(next_node)
            dist_so_far += edge_len
            current = next_node

        return route

    def build_solution(self):
        return [self.build_random_route() for _ in range(self.config.num_vehicles)]

    def fitness(self, solution):
        covered_edges = set()
        edge_usage = {}

        for route in solution:
            for u, v in zip(route[:-1], route[1:]):
                if (u, v) in self.edge_length_cache:
                    edge_usage[(u, v)] = edge_usage.get((u, v), 0) + 1
                    covered_edges.add((u, v))
                elif (v, u) in self.edge_length_cache:
                    edge_usage[(v, u)] = edge_usage.get((v, u), 0) + 1
                    covered_edges.add((v, u))

        coverage_reward = sum(self.config.coverage_factor * self.edge_length_cache[e] for e in covered_edges)
        overlap_penalty = sum(self.config.overlap_penalty_factor * (count - 3) * self.edge_length_cache[e]
                              for e, count in edge_usage.items() if count > 3)
        return coverage_reward - overlap_penalty




    def map_to_bin(self, coverage, overlap):
        cov_bin = np.digitize(coverage, self.config.coverage_bins) - 1
        cov_bin = max(0, min(cov_bin, len(self.config.coverage_bins) - 2))
        ovl_bin = np.digitize(overlap, self.config.overlap_bins) - 1
        ovl_bin = max(0, min(ovl_bin, len(self.config.overlap_bins) - 2))
        return (cov_bin, ovl_bin)

    def initialize_archive(self):
        for _ in range(self.config.initial_population_size):
            sol = self.build_solution()
            fit = self.fitness(sol)
            coverage, overlap = self.calculate_descriptors(sol)
            cov_bin, ovl_bin = self.map_to_bin(coverage, overlap)
            key = (cov_bin, ovl_bin)
            if key not in self.archive or fit > self.archive[key]['fitness']:
                self.archive[key] = {'solution': sol, 'fitness': fit}

    def mutate(self, solution):
        new_sol = []
        for route in solution:
            if random.random() < self.config.mutation_rate and len(route) > 4:
                i = random.randint(0, len(route)-2)
                j = random.randint(i+1, len(route)-1)
                sub_start, sub_end = route[i], route[j]
                sub_route = self._build_subroute(sub_start, sub_end)
                new_route = route[:i+1] + sub_route[1:-1] + route[j:]
                new_sol.append(new_route)
            else:
                new_sol.append(route)
        return new_sol

    def _build_subroute(self, start, end):
        route = [start]
        current = start
        for _ in range(20):
            neighbors = self.adjacency_cache[current]
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            route.append(next_node)
            current = next_node
            if current == end:
                break
        # Only append end if there's a direct edge from current to end
        if current != end and end in self.adjacency_cache.get(current, []):
            route.append(end)
        return route

    def calculate_descriptors(self, solution):
        edge_counts = {}
        covered_edges = set()
        for route in solution:
            for u, v in zip(route[:-1], route[1:]):
                # Check both directions and handle missing edges
                if (u, v) in self.edge_length_cache:
                    edge = (u, v)
                elif (v, u) in self.edge_length_cache:
                    edge = (v, u)
                else:
                    continue  # Skip invalid edges
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
                covered_edges.add(edge)
        
        total_coverage = sum(self.edge_length_cache[e] * count for e, count in edge_counts.items() if e in self.edge_length_cache)
        total_overlap = sum(self.edge_length_cache[e] * (count - 3) for e, count in edge_counts.items() if count > 3 and e in self.edge_length_cache)
        return total_coverage, total_overlap
    def run(self):
        self.initialize_archive()

        for _ in range(self.config.num_iterations):
            if not self.archive:
                sol = self.build_solution()
                fit = self.fitness(sol)
                coverage, overlap = self.calculate_descriptors(sol)
                cov_bin, ovl_bin = self.map_to_bin(coverage, overlap)
                self.archive[(cov_bin, ovl_bin)] = {'solution': sol, 'fitness': fit}
                continue

            key = random.choice(list(self.archive.keys()))
            parent = self.archive[key]['solution']
            offspring = self.mutate(parent)
            offspring_fit = self.fitness(offspring)
            coverage, overlap = self.calculate_descriptors(offspring)
            cov_bin, ovl_bin = self.map_to_bin(coverage, overlap)
            offspring_key = (cov_bin, ovl_bin)

            if offspring_key not in self.archive or offspring_fit > self.archive.get(offspring_key, {'fitness': -float('inf')})['fitness']:
                self.archive[offspring_key] = {'solution': offspring, 'fitness': offspring_fit}

        best_fitness = -float('inf')
        best_solution = None
        for cell in self.archive.values():
            if cell['fitness'] > best_fitness:
                best_fitness = cell['fitness']
                best_solution = cell['solution']
        return best_solution, best_fitness

###############################################################################
# 5) CONVERT SOLUTION TO GEOJSON
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
            if (u, v) in G.edges():
                edge_data = G.get_edge_data(u, v)
                geom = edge_data[0]['geometry'] if 'geometry' in edge_data[0] else LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
                lines.append(geom)
            elif (v, u) in G.edges():
                edge_data = G.get_edge_data(v, u)
                geom = edge_data[0]['geometry'] if 'geometry' in edge_data[0] else LineString([(G.nodes[v]['x'], G.nodes[v]['y']), (G.nodes[u]['x'], G.nodes[u]['y'])])
                lines.append(geom)
        if lines:
            color = colors[vid % len(colors)]
            feature = {
                "type": "Feature",
                "properties": {"vehicle": vid + 1, "stroke": color, "stroke-width": 3, "stroke-opacity": 0.8},
                "geometry": {"type": "MultiLineString", "coordinates": [list(line.coords) for line in lines]}
            }
            features.append(feature)
    return {"type": "FeatureCollection", "features": features}

###############################################################################
# 6) STREAMLIT MAIN APP
###############################################################################
def main():
    st.set_page_config(layout="wide")
    st.title("Multi-Vehicle Coverage (MAP-Elites)")

    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles", 2, 50, 50)
        initial_pop_size = st.slider("Initial Population Size", 5, 50, 20)
        num_iterations = st.slider("Iterations", 10, 200, 50)
        coverage_factor = st.slider("Coverage Factor", 1000, 20000, 5000, step=1000)
        overlap_penalty = st.slider("Overlap Penalty Factor", 1000, 50000, 10000, step=1000)
        run_btn = st.button("Run MAP-Elites")

    if run_btn:
        start_time = time.time()
        config = MAPElitesConfig()
        config.num_vehicles = num_vehicles
        config.initial_population_size = initial_pop_size
        config.num_iterations = num_iterations
        config.coverage_factor = coverage_factor
        config.overlap_penalty_factor = overlap_penalty

        with st.spinner("Loading road network..."):
            G = load_road_network("Coimbatore, India")

        with st.spinner("Fetching police stations..."):
            police_nodes = get_police_station_nodes(G)

        with st.spinner("Running MAP-Elites..."):
            optimizer = MAPElitesOptimizer(G, config, police_nodes)
            best_sol, best_fit = optimizer.run()

        geojson = solution_to_geojson(G, best_sol)
        m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
        boundary_gdf = ox.geocode_to_gdf("Coimbatore, India")
        m.add_gdf(boundary_gdf, layer_name="Boundary", style={"color": "#FF0000", "fillOpacity": 0.1})
        m.add_geojson(geojson, layer_name="Routes")
        st.success(f"Best Fitness: {best_fit:.2f}")
        m.to_streamlit()
        st.write(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()