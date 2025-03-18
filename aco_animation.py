import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import random
import numpy as np
from shapely.geometry import LineString

# Configure OSMnx caching and logging
ox.settings.use_cache = True
ox.settings.log_console = True

###############################################################################
# 1) LOAD / CACHING THE ROAD NETWORK
###############################################################################
def load_road_network(place="Coimbatore, India", cache_dir="road_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    safe_place = place.replace(" ", "_").replace(",", "")
    cache_file = os.path.join(cache_dir, f"{safe_place}_network.graphml")
    
    # If you have a local "major roads" file:
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
# 2) GET POLICE STATION NODES (Fixed Starting Points)
###############################################################################
def get_police_station_nodes(G, place="Coimbatore, India"):
    try:
        gdf_police = ox.features_from_place(place, tags={"amenity": "police"})
    except Exception as e:
        st.warning(f"Could not fetch police stations: {e}")
        return []
    if gdf_police.empty:
        st.info("No police stations found. Falling back to random nodes.")
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
# 3) PSO CONFIGURATION
###############################################################################
class PSOConfig:
    def __init__(self):
        self.num_vehicles = 5          # Vehicles per solution
        self.swarm_size = 20           # Number of particles
        self.num_iterations = 50

        # PSO parameters (conceptual for route-based updates)
        self.w = 0.5   # inertia (not used directly here)
        self.c1 = 1.5  # cognitive factor
        self.c2 = 1.5  # social factor

        # Scaled reward/penalty factors (lower values for easy comparison)
        self.coverage_factor = 1       # reward per meter of covered edge
        self.overlap_penalty_factor = 2  # penalty per meter for edges used >3 times

        # Route-building constraints
        self.max_route_distance = 15000  # maximum distance per route (in meters)
        self.mutation_rate = 0.5         # increased chance for mutation to encourage exploration

###############################################################################
# 4) PSO ROUTE OPTIMIZER (USING POLICE STATION STARTS)
###############################################################################
class PSORouteOptimizer:
    def __init__(self, G, config, police_nodes=None):
        self.G = G
        self.config = config
        # Fixed starting points from police stations (if available)
        self.starting_nodes = police_nodes if police_nodes and len(police_nodes) > 0 else None

        # Build caches for adjacency and edge lengths
        self.adjacency_cache = {node: list(self.G.successors(node)) for node in self.G.nodes()}
        self.edge_length_cache = {}
        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 0)
            self.edge_length_cache[(u, v)] = length
        self.all_edges = set(self.edge_length_cache.keys())

        # Swarm structures
        self.swarm = []               # current solutions (list of solutions)
        self.personal_best = []       # best solution per particle
        self.personal_best_fitness = []
        self.global_best = None
        self.global_best_fitness = -float('inf')

    # Build a random route; if fixed starting node(s) exist, use them (round-robin)
    def build_random_route(self, vehicle_index=0):
        if self.starting_nodes:
            start = self.starting_nodes[vehicle_index % len(self.starting_nodes)]
        else:
            nodes = list(self.G.nodes())
            start = random.choice(nodes)
        route = [start]
        dist_so_far = 0.0
        current = start
        while True:
            neighbors = self.adjacency_cache.get(current, [])
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

    # Build a complete solution: one route per vehicle
    def build_solution(self):
        solution = []
        for i in range(self.config.num_vehicles):
            solution.append(self.build_random_route(i))
        return solution

    # Fitness: reward for coverage minus penalty for overlap
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
        overlap_penalty = 0.0
        for e, count in edge_usage.items():
            if count > 3:
                overlap_penalty += self.config.overlap_penalty_factor * (count - 3) * self.edge_length_cache[e]
        return coverage_reward - overlap_penalty

    # Initialize the swarm
    def initialize_swarm(self):
        self.swarm = []
        self.personal_best = []
        self.personal_best_fitness = []
        self.global_best = None
        self.global_best_fitness = -float('inf')
        for _ in range(self.config.swarm_size):
            sol = self.build_solution()
            fit = self.fitness(sol)
            self.swarm.append(sol)
            self.personal_best.append(sol)
            self.personal_best_fitness.append(fit)
            if fit > self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best = sol

    # Velocity update: for each vehicle route, choose from current, personal best, or global best
    def velocity_update(self, i, current_sol):
        pbest = self.personal_best[i]
        gbest = self.global_best
        new_sol = []
        for v in range(self.config.num_vehicles):
            r = random.random()
            if r < 0.34:
                new_sol.append(current_sol[v])
            elif r < 0.67:
                new_sol.append(pbest[v])
            else:
                new_sol.append(gbest[v])
        return new_sol

    # Mutation: randomly modify route segments
    def mutate(self, solution):
        new_sol = []
        for route in solution:
            if random.random() < self.config.mutation_rate and len(route) > 4:
                i = random.randint(0, len(route) - 2)
                j = random.randint(i + 1, len(route) - 1)
                sub_start = route[i]
                sub_end = route[j]
                sub_route = self._build_subroute(sub_start, sub_end)
                mutated_route = route[:i+1] + sub_route[1:-1] + route[j:]
                new_sol.append(mutated_route)
            else:
                new_sol.append(route)
        return new_sol

    def _build_subroute(self, start_node, end_node):
        route = [start_node]
        current = start_node
        steps = 0
        while current != end_node and steps < 20:
            neighbors = self.adjacency_cache.get(current, [])
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            route.append(next_node)
            current = next_node
            steps += 1
            if current == end_node:
                break
        if route[-1] != end_node:
            route.append(end_node)
        return route

    # Main PSO loop
    def run(self):
        self.initialize_swarm()
        for iteration in range(self.config.num_iterations):
            for i in range(self.config.swarm_size):
                candidate = self.velocity_update(i, self.swarm[i])
                mutated = self.mutate(candidate)
                new_fit = self.fitness(mutated)
                old_fit = self.fitness(self.swarm[i])
                if new_fit > old_fit:
                    self.swarm[i] = mutated
                    if new_fit > self.personal_best_fitness[i]:
                        self.personal_best[i] = mutated
                        self.personal_best_fitness[i] = new_fit
                        if new_fit > self.global_best_fitness:
                            self.global_best = mutated
                            self.global_best_fitness = new_fit
        return self.global_best, self.global_best_fitness

###############################################################################
# 5) CONVERT SOLUTION TO GEOJSON FOR VISUALIZATION (WITH BACKGROUND MAP)
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
                if edge_data is not None:
                    edge_dict = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
                    geom = edge_dict.get('geometry')
                    if geom is None:
                        geom = LineString([
                            (G.nodes[u]['x'], G.nodes[u]['y']),
                            (G.nodes[v]['x'], G.nodes[v]['y'])
                        ])
                    lines.append(geom)
            elif (v, u) in G.edges():
                edge_data = G.get_edge_data(v, u)
                if edge_data is not None:
                    edge_dict = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
                    geom = edge_dict.get('geometry')
                    if geom is None:
                        geom = LineString([
                            (G.nodes[v]['x'], G.nodes[v]['y']),
                            (G.nodes[u]['x'], G.nodes[u]['y'])
                        ])
                    lines.append(geom)
        if lines:
            color = colors[vid % len(colors)]
            feature = {
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
            }
            features.append(feature)
    return {"type": "FeatureCollection", "features": features}

###############################################################################
# 6) STREAMLIT MAIN APP
###############################################################################
def main():
    st.set_page_config(layout="wide")
    st.title("Multi-Vehicle Coverage (PSO with Fixed Police Station Starts)")

    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles", 2, 50, 5)
        swarm_size = st.slider("Swarm Size", 5, 50, 20)
        num_iterations = st.slider("Iterations", 10, 200, 50)
        coverage_factor = st.slider("Coverage Factor", 1, 10, 1, step=1,
                                    help="Reward factor per meter for each covered edge")
        overlap_penalty_factor = st.slider("Overlap Penalty Factor", 1, 10, 2, step=1,
                                           help="Penalty factor per meter for edges used >3 times")
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.5, step=0.1)
        run_btn = st.button("Run PSO Optimization")

    if run_btn:
        config = PSOConfig()
        config.num_vehicles = num_vehicles
        config.swarm_size = swarm_size
        config.num_iterations = num_iterations
        config.coverage_factor = coverage_factor
        config.overlap_penalty_factor = overlap_penalty_factor
        config.mutation_rate = mutation_rate

        with st.spinner("Loading road network..."):
            G = load_road_network("Coimbatore, India")
        with st.spinner("Fetching police station nodes..."):
            police_nodes = get_police_station_nodes(G, "Coimbatore, India")
            st.write(f"Found {len(police_nodes)} police station node(s).")
        # Use police station nodes as fixed starting points.
        optimizer = PSORouteOptimizer(G, config, police_nodes)
        with st.spinner("Running PSO Optimization..."):
            best_solution, best_fit = optimizer.run()

        st.success(f"Optimization complete. Best Fitness: {best_fit:.2f}")
        geojson_data = solution_to_geojson(G, best_solution)

        # Create a map with background: show city boundary and overlay the PSO routes.
        m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
        boundary_gdf = ox.geocode_to_gdf("Coimbatore, India")
        m.add_gdf(boundary_gdf, layer_name="Coimbatore Boundary",
                  style={"color": "#FF0000", "fillOpacity": 0.1})
        m.add_geojson(geojson_data, layer_name="PSO Routes")

        st.write("Fixed starting points (police stations) were used for each vehicle and remain unchanged.")
        m.to_streamlit()

if __name__ == "__main__":
    main()
