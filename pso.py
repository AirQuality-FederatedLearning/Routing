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

random.seed(42)
np.random.seed(42)

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

    # If you have a local "major roads" .graphml:
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
# 3) PSO CONFIG
###############################################################################
class PSOConfig:
    def __init__(self):
        # Basic swarm config
        self.num_vehicles = 5          # Vehicles per solution
        self.swarm_size = 20          # Number of particles
        self.num_iterations = 50

        # PSO parameters
        # We'll do a simplistic route-based "velocity" approach, so these might be used only conceptually
        self.w = 0.5   # inertia (not heavily used in discrete route approach)
        self.c1 = 1.5  # cognitive factor
        self.c2 = 1.5  # social factor

        # Coverage / overlap factors
        self.coverage_factor = 5000
        self.overlap_penalty_factor = 10000

        # Route-building / mutation
        self.max_route_distance = 15000  # if you want to limit route length
        self.mutation_rate = 0.2         # chance to mutate a route

###############################################################################
# 4) PSO OPTIMIZER (ROUTE-BASED)
###############################################################################
class PSORouteOptimizer:
    def __init__(self, G, config, start_nodes=None):
        self.G = G
        self.config = config
        self.start_nodes = start_nodes

        # Build adjacency + edge length caches
        self.adjacency_cache = {}
        self.edge_length_cache = {}

        for node in self.G.nodes():
            # For a directed graph, successors might differ from neighbors
            self.adjacency_cache[node] = list(self.G.successors(node))

        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 0)
            self.edge_length_cache[(u, v)] = length

        # For coverage tracking
        self.all_edges = set(self.edge_length_cache.keys())

        # Swarm structures
        self.swarm = []               # current solutions
        self.personal_best = []       # personal best solutions
        self.personal_best_fitness = []
        self.global_best = None
        self.global_best_fitness = -float('inf')

    # ------------------- ROUTE BUILDING -------------------
    def build_random_route(self):
        """
        Build a single random route from a random node, walking until
        we exceed max_route_distance or run out of successors.
        """
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
        """
        A solution is a list of routes, one per vehicle.
        """
        return [self.build_random_route() for _ in range(self.config.num_vehicles)]

    # ------------------- FITNESS -------------------
    def fitness(self, solution):
        """
        Coverage-based fitness:
         coverage_reward = coverage_factor * sum(edge_length for newly covered edges)
         overlap_penalty if more than 3 vehicles use the same edge
        """
        covered_edges = set()
        edge_usage = {}

        for route in solution:
            for u, v in zip(route[:-1], route[1:]):
                if (u, v) in self.edge_length_cache:
                    edge_usage[(u, v)] = edge_usage.get((u, v), 0) + 1
                    covered_edges.add((u, v))
                elif (v, u) in self.edge_length_cache:
                    # handle the reverse if the graph is directed
                    edge_usage[(v, u)] = edge_usage.get((v, u), 0) + 1
                    covered_edges.add((v, u))

        # coverage reward
        coverage_reward = 0.0
        for e in covered_edges:
            coverage_reward += self.config.coverage_factor * self.edge_length_cache[e]

        # overlap penalty
        overlap_penalty = 0.0
        for e, count in edge_usage.items():
            if count > 3:
                overlap_penalty += self.config.overlap_penalty_factor * (count - 3) * self.edge_length_cache[e]

        return coverage_reward - overlap_penalty

    # ------------------- INITIALIZE SWARM -------------------
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

    # ------------------- VELOCITY UPDATE -------------------
    def velocity_update(self, i, current_sol):
        """
        For route-based solutions, we define a discrete 'velocity' by
        partial adoption of personal best and global best routes.

        We'll create a new solution by, for each vehicle route,
        randomly picking from (old route, personal best route, global best route).
        """
        pbest_sol = self.personal_best[i]
        gbest_sol = self.global_best
        new_sol = []

        for vehicle_idx in range(self.config.num_vehicles):
            r = random.random()
            if r < 0.34:
                # keep old route
                new_sol.append(current_sol[vehicle_idx])
            elif r < 0.67:
                # copy from personal best
                new_sol.append(pbest_sol[vehicle_idx])
            else:
                # copy from global best
                new_sol.append(gbest_sol[vehicle_idx])

        return new_sol

    # ------------------- MUTATION -------------------
    def mutate(self, solution):
        """
        With mutation_rate probability, pick a route in the solution,
        pick random indices, and re-route using a shortest path or new random sub-route.

        For demonstration, let's do a simple approach:
         - For each vehicle route, with chance 'mutation_rate', we pick two random nodes
           in the route and attempt to replace that segment with a new random route.
        """
        new_sol = []
        for route in solution:
            if random.random() < self.config.mutation_rate and len(route) > 4:
                i = random.randint(0, len(route)-2)
                j = random.randint(i+1, len(route)-1)
                # new random route from route[i] to route[j]
                sub_start = route[i]
                sub_end   = route[j]

                # build a small random route from sub_start to sub_end
                sub_route = self._build_subroute(sub_start, sub_end)
                mutated_route = route[:i+1] + sub_route[1:-1] + route[j:]
                new_sol.append(mutated_route)
            else:
                new_sol.append(route)
        return new_sol

    def _build_subroute(self, start_node, end_node):
        """
        Build a simple random route between start_node and end_node if possible.
        We'll do a random walk for up to 20 steps trying to reach end_node or get close.
        """
        route = [start_node]
        current = start_node
        steps = 0
        while current != end_node and steps < 20:
            neighbors = self.adjacency_cache[current]
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            route.append(next_node)
            current = next_node
            steps += 1
            if current == end_node:
                break

        if route[-1] != end_node:
            # just append end_node at the end for demonstration
            route.append(end_node)
        return route

    # ------------------- MAIN PSO LOOP -------------------
    def run(self):
        self.initialize_swarm()

        for iteration in range(self.config.num_iterations):
            for i in range(self.config.swarm_size):
                # Velocity update: build new candidate
                candidate = self.velocity_update(i, self.swarm[i])
                # Mutation
                mutated = self.mutate(candidate)

                # Evaluate
                new_fit = self.fitness(mutated)
                old_fit = self.fitness(self.swarm[i])

                # Accept if improved
                if new_fit > old_fit:
                    self.swarm[i] = mutated

                    # Update personal best
                    if new_fit > self.personal_best_fitness[i]:
                        self.personal_best[i] = mutated
                        self.personal_best_fitness[i] = new_fit

                        # Check global best
                        if new_fit > self.global_best_fitness:
                            self.global_best = mutated
                            self.global_best_fitness = new_fit

        return self.global_best, self.global_best_fitness

###############################################################################
# 5) CONVERT SOLUTION TO GEOJSON FOR VISUALIZATION
###############################################################################
def solution_to_geojson(G, solution):
    """
    Convert a solution (list of routes) into a GeoJSON FeatureCollection.
    Each route is assigned a unique color.
    """
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
            # handle directed edges
            if (u, v) in G.edges():
                edge_data = G.get_edge_data(u, v)
                if edge_data is not None:
                    # if multi-edge
                    edge_dict = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
                    geom = edge_dict.get('geometry', None)
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
                    geom = edge_dict.get('geometry', None)
                    if geom is None:
                        geom = LineString([
                            (G.nodes[v]['x'], G.nodes[v]['y']),
                            (G.nodes[u]['x'], G.nodes[u]['y'])
                        ])
                    lines.append(geom)
            # else no direct edge, skip

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
    st.title("Multi-Vehicle Coverage (PSO - No RRT)")

    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles", 2, 50, 50)
        swarm_size = st.slider("Swarm Size", 5, 50, 20)
        num_iterations = st.slider("Iterations", 10, 200, 50)

        coverage_factor = st.slider("Coverage Factor", 1000, 20000, 5000, step=1000,
                                    help="Reward factor for each covered edge * length")
        overlap_penalty_factor = st.slider("Overlap Penalty Factor", 1000, 50000, 10000, step=1000,
                                           help="Penalty factor for edges used >3 times.")
        run_btn = st.button("Run PSO")

    if run_btn:
        start_time = time.time()
        # Build config
        config = PSOConfig()
        config.num_vehicles = num_vehicles
        config.swarm_size = swarm_size
        config.num_iterations = num_iterations
        config.coverage_factor = coverage_factor
        config.overlap_penalty_factor = overlap_penalty_factor

        # Load graph
        with st.spinner("Loading road network..."):
            G = load_road_network("Coimbatore, India")

        # Fetch police stations and get their nodes
        with st.spinner("Fetching police station nodes..."):
            police_station_nodes = get_police_station_nodes(G, place="Coimbatore, India")

        # Run PSO
        with st.spinner("Running PSO..."):
            optimizer = PSORouteOptimizer(G, config, start_nodes=police_station_nodes)
            best_solution, best_fit = optimizer.run()

        # Convert best solution to GeoJSON and show on map
        geojson_data = solution_to_geojson(G, best_solution)
        m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
        # Optionally add the city boundary
        boundary_gdf = ox.geocode_to_gdf("Coimbatore, India")
        m.add_gdf(boundary_gdf, layer_name="Coimbatore Boundary",
                  style={"color": "#FF0000", "fillOpacity": 0.1})
        m.add_geojson(geojson_data, layer_name="PSO Routes")

        st.success(f"Best Fitness: {best_fit:.2f}")
        m.to_streamlit()

        end_time = time.time()
        st.write(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
