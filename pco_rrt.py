import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import random
import numpy as np
from shapely.geometry import LineString
from tqdm import tqdm
import time
import json

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

    # Example fallback to major roads if you have it locally
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
# 3) PSO + RRT CONFIG
###############################################################################
class PSOConfig:
    def __init__(self):
        self.num_vehicles = 5          # Vehicles
        self.swarm_size = 20          # Particles
        self.num_iterations = 50

        # PSO parameters (conceptual for discrete routes)
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5

        # Coverage / overlap
        self.coverage_factor = 5000
        self.overlap_penalty_factor = 10000

        # RRT-like expansions
        self.max_route_steps = 20      # how many expansions we do per route
        self.max_route_distance = 15000  # do not exceed total route distance
        self.explore_prob = 0.3        # not heavily used here, but can be for random picking
        self.mutation_rate = 0.2       # route mutation probability

###############################################################################
# 4) PSO + RRT WITH PATH CONNECTIVITY
###############################################################################
class PSORRTConnectivityOptimizer:
    def __init__(self, G, config, start_nodes=None):
        self.G = G
        self.config = config
        self.start_nodes = start_nodes

        # Build adjacency + edge length caches
        self.edge_length_cache = {}
        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 0)
            self.edge_length_cache[(u, v)] = length

        # Keep a set of all directed edges for coverage
        self.all_edges = set(self.edge_length_cache.keys())

        # Initialize swarm structures
        self.swarm = []
        self.personal_best = []
        self.personal_best_fitness = []
        self.global_best = None
        self.global_best_fitness = -float('inf')

    def build_rrt_route(self):
        """
        Build a single route using a 'random node' approach but 
        ensure connectivity by appending the shortest path from 
        current -> random node if possible, stopping if route distance 
        would exceed max_route_distance.
        """
        nodes_list = list(self.G.nodes())
        if not nodes_list:
            return []

        # Pick a random start node from the given list (police stations)
        current = random.choice(self.start_nodes) if self.start_nodes else random.choice(nodes_list)
        route = [current]
        dist_so_far = 0.0

        for _ in range(self.config.max_route_steps):
            rand_node = random.choice(nodes_list)
            if rand_node == current:
                continue

            # Attempt shortest path from current to rand_node
            try:
                sp = nx.shortest_path(self.G, current, rand_node, weight='length')
            except nx.NetworkXNoPath:
                # no path, skip
                continue

            # measure subpath length
            subpath_len = 0.0
            for u, v in zip(sp[:-1], sp[1:]):
                # check forward or backward in edge_length_cache
                if (u, v) in self.edge_length_cache:
                    subpath_len += self.edge_length_cache[(u, v)]
                elif (v, u) in self.edge_length_cache:
                    subpath_len += self.edge_length_cache[(v, u)]

            # check if adding this subpath would exceed route distance
            if dist_so_far + subpath_len > self.config.max_route_distance:
                break

            # Append subpath (minus the repeated current node) to route
            for node_idx in range(1, len(sp)):
                route.append(sp[node_idx])

            dist_so_far += subpath_len
            current = rand_node

        return route

    def build_solution(self):
        """
        A solution = list of routes, one per vehicle.
        """
        return [self.build_rrt_route() for _ in range(self.config.num_vehicles)]

    def fitness(self, solution):
        """
        Coverage-based fitness:
          coverage_reward = coverage_factor * sum(edge_length for covered edges)
          overlap_penalty if more than 3 vehicles on same edge
          final = coverage_reward - overlap_penalty
        """
        covered_edges = set()
        edge_usage = {}

        for route in tqdm(solution,desc="Calculating fitness"):
            for u, v in zip(route[:-1], route[1:]):
                if (u, v) in self.edge_length_cache:
                    edge_usage[(u, v)] = edge_usage.get((u, v), 0) + 1
                    covered_edges.add((u, v))
                elif (v, u) in self.edge_length_cache:
                    edge_usage[(v, u)] = edge_usage.get((v, u), 0) + 1
                    covered_edges.add((v, u))

        # coverage reward
        coverage_reward = 0
        for e in covered_edges:
            coverage_reward += self.config.coverage_factor * self.edge_length_cache[e]

        # overlap penalty
        overlap_penalty = 0
        for e, count in edge_usage.items():
            if count > 3:
                overlap_penalty += self.config.overlap_penalty_factor * (count - 3) * self.edge_length_cache[e]

        return coverage_reward - overlap_penalty

    def initialize_swarm(self):
        self.swarm.clear()
        self.personal_best.clear()
        self.personal_best_fitness.clear()
        self.global_best = None
        self.global_best_fitness = -float('inf')

        for _ in range(self.config.swarm_size):
            sol = self.build_solution()
            fit = self.fitness(sol)
            self.swarm.append(sol)
            self.personal_best.append(sol)
            self.personal_best_fitness.append(fit)
            if fit > self.global_best_fitness:
                self.global_best = sol
                self.global_best_fitness = fit

    def velocity_update(self, i, sol):
        """
        For route-based solutions, define a discrete 'velocity' by 
        picking each route from (old solution, personal best, global best).
        """
        pbest = self.personal_best[i]
        gbest = self.global_best
        new_sol = []

        for vehicle_idx in range(self.config.num_vehicles):
            r = random.random()
            if r < 0.34:
                new_sol.append(sol[vehicle_idx])      # keep old route
            elif r < 0.67:
                new_sol.append(pbest[vehicle_idx])   # from personal best
            else:
                new_sol.append(gbest[vehicle_idx])   # from global best

        return new_sol

    def mutate(self, sol):
        """
        With 'mutation_rate' probability, pick one route and 
        rebuild part of it using a new RRT path.
        """
        new_sol = []
        for route in tqdm(sol,desc="Mutating"):
            if random.random() < self.config.mutation_rate:
                # pick random index to 'cut' the route
                if len(route) > 3:
                    i = random.randint(1, len(route)-2)
                    # from route[i], build a new partial path
                    prefix = route[:i]
                    partial_route = self.build_rrt_route()
                    # merge
                    mutated = prefix + partial_route
                    new_sol.append(mutated)
                else:
                    new_sol.append(route)
            else:
                new_sol.append(route)
        return new_sol

    def run(self):
        self.initialize_swarm()

        for it in range(self.config.num_iterations):
            for i in range(self.config.swarm_size):
                # velocity update
                candidate = self.velocity_update(i, self.swarm[i])
                # mutation
                mutated = self.mutate(candidate)

                # evaluate new
                new_fit = self.fitness(mutated)
                old_fit = self.fitness(self.swarm[i])

                # accept if improved
                if new_fit > old_fit:
                    self.swarm[i] = mutated

                    # update personal best
                    if new_fit > self.personal_best_fitness[i]:
                        self.personal_best[i] = mutated
                        self.personal_best_fitness[i] = new_fit

                        # update global best
                        if new_fit > self.global_best_fitness:
                            self.global_best = mutated
                            self.global_best_fitness = new_fit

        return self.global_best, self.global_best_fitness

###############################################################################
# 5) CONVERT SOLUTION TO GEOJSON
###############################################################################
def solution_to_geojson(G, solution):
    """
    Convert solution (list of routes) => GeoJSON FeatureCollection.
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
        for u, v in tqdm(zip(route[:-1], route[1:]),desc="Building GeoJSON"):
            if (u, v) in G.edges():
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    # handle multi-edge
                    edge_dict = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
                    geom = edge_dict.get("geometry", None)
                    if geom is None:
                        geom = LineString([ 
                            (G.nodes[u]["x"], G.nodes[u]["y"]),
                            (G.nodes[v]["x"], G.nodes[v]["y"]),
                        ])
                    lines.append(geom)
            elif (v, u) in G.edges():
                edge_data = G.get_edge_data(v, u)
                if edge_data:
                    edge_dict = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
                    geom = edge_dict.get("geometry", None)
                    if geom is None:
                        geom = LineString([ 
                            (G.nodes[v]["x"], G.nodes[v]["y"]),
                            (G.nodes[u]["x"], G.nodes[u]["y"]),
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
    st.title("PSO + RRT (Path Connectivity) Coverage Demo")

    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles", 2, 50, 25)
        swarm_size = st.slider("Swarm Size", 5, 50, 20)
        num_iterations = st.slider("Iterations", 10, 200, 50)

        coverage_factor = st.slider("Coverage Factor", 1000, 20000, 5000, step=1000,
                                    help="Reward factor for each newly covered edge * length.")
        overlap_penalty_factor = st.slider("Overlap Penalty Factor", 1000, 50000, 10000, step=1000,
                                           help="Penalty factor if >3 vehicles reuse an edge.")
        max_route_distance = st.slider("Max Route Distance", 5000, 50000, 15000, step=5000,
                                       help="Stops building the route if we exceed this.")
        max_route_steps = st.slider("Max Route Steps", 5, 50, 20,
                                    help="How many random expansions to attempt in building a route.")
        run_btn = st.button("Run PSO-RRT Connectivity")

    if run_btn:
        # Build config
        start_time = time.time()
        config = PSOConfig()
        config.num_vehicles = num_vehicles
        config.swarm_size = swarm_size
        config.num_iterations = num_iterations
        config.coverage_factor = coverage_factor
        config.overlap_penalty_factor = overlap_penalty_factor
        config.max_route_distance = max_route_distance
        config.max_route_steps = max_route_steps

        with st.spinner("Loading road network..."):
            G = load_road_network("Coimbatore, India")

        # Fetch police stations and get their nodes
        with st.spinner("Fetching police station nodes..."):
            police_station_nodes = get_police_station_nodes(G, place="Coimbatore, India")

        with st.spinner("Running PSO + RRT expansions..."):
            optimizer = PSORRTConnectivityOptimizer(G, config, start_nodes=police_station_nodes)
            best_solution, best_fit = optimizer.run()

        geojson_data = solution_to_geojson(G, best_solution)
        with open("output.geojson", "w") as f:
            json.dump(geojson_data, f, indent=2)
        # Show in Leafmap
        m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
        boundary_gdf = ox.geocode_to_gdf("Coimbatore, India")
        m.add_gdf(boundary_gdf, layer_name="Coimbatore Boundary",
                  style={"color": "#FF0000", "fillOpacity": 0.1})
        m.add_geojson(geojson_data, layer_name="PSO-RRT-Connectivity")

        st.success(f"Best Fitness Achieved: {best_fit:.2f}")
        m.to_streamlit()
        end_time = time.time()
        st.write(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
