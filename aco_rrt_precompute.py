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
import matplotlib.pyplot as plt


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
    Fetch police stations from OSM and return:
    - Nearest road network nodes (for routing)
    - Original police station geometries (for visualization)
    """
    try:
        gdf_police = ox.features_from_place(place, tags={"amenity": "police"})
    except Exception as e:
        st.warning(f"Could not fetch police stations: {e}")
        return [], None

    if gdf_police.empty:
        st.info("No police stations found with the given tag.")
        return [], None

    station_nodes = []
    valid_police_gdf = gdf_police[~gdf_police.geometry.is_empty].copy()
    valid_police_gdf['geometry'] = valid_police_gdf.geometry.centroid  # Ensure points

    for idx, row in valid_police_gdf.iterrows():
        point = row.geometry
        node_id = ox.distance.nearest_nodes(G, point.x, point.y)
        station_nodes.append(node_id)
    
    return list(set(station_nodes)), valid_police_gdf

###############################################################################
# 3) SIMPLE RRT-STYLE PRECOMPUTED SOLUTIONS
###############################################################################
def build_rrt_route(G, max_route_distance=15000, max_route_steps=20):
    """
    Build a single route using an RRT-like approach:
    - Start from a random node
    - At each step, pick a random node in the graph and
      append the shortest path from 'current' to that node if feasible
    - Stop if total distance exceeds max_route_distance or we hit max_route_steps
    """
    nodes_list = list(G.nodes())
    if not nodes_list:
        return []

    current = random.choice(nodes_list)
    route = [current]
    dist_so_far = 0.0

    edge_length_cache = {}
    for u, v, data in G.edges(data=True):
        length = data.get('length', 0)
        edge_length_cache[(u, v)] = length

    for _ in range(max_route_steps):
        rand_node = random.choice(nodes_list)
        if rand_node == current:
            continue

        # Try shortest path
        try:
            sp = nx.shortest_path(G, current, rand_node, weight='length')
        except nx.NetworkXNoPath:
            continue

        # measure subpath length
        sub_len = 0
        for u, v in zip(sp[:-1], sp[1:]):
            if (u, v) in edge_length_cache: 
                sub_len += edge_length_cache[(u, v)]
            elif (v, u) in edge_length_cache:
                sub_len += edge_length_cache[(v, u)]

        if dist_so_far + sub_len > max_route_distance:
            break

        # append the subpath
        for node_idx in range(1, len(sp)):
            route.append(sp[node_idx])
        dist_so_far += sub_len
        current = rand_node

    return route

def build_rrt_solution(G, num_vehicles, max_route_distance=15000, max_route_steps=20):
    """
    Build a solution (list of routes), one route per vehicle, using RRT expansions.
    """
    return [
        build_rrt_route(G, max_route_distance, max_route_steps) 
        for _ in range(num_vehicles)
    ]

def generate_precomputed_solutions(
    G, 
    how_many=3, 
    num_vehicles=5, 
    max_route_distance=15000, 
    max_route_steps=20
):
    """
    Generate `how_many` solutions using RRT expansions.
    Return a list of solutions. 
    Each solution is a list of routes (one route per vehicle).
    """
    solutions = []
    for _ in tqdm(range(how_many), desc="Generating RRT Solutions"):
        sol = build_rrt_solution(G, num_vehicles, max_route_distance, max_route_steps)
        solutions.append(sol)
    return solutions

###############################################################################
# 4) ACO CONFIG & OPTIMIZER
###############################################################################
class ACOConfig:
    def __init__(self):
        self.num_ants = 5            # also equals number of vehicles
        self.num_iterations = 50
        self.alpha = 1.0            # pheromone influence
        self.beta = 2.0             # distance influence
        self.evaporation_rate = 0.1
        self.Q = 100
        self.max_route_distance = 15000
        self.penalty_factor = 10000      # if overlap > 3 vehicles
        self.coverage_factor = 5000      # coverage reward factor
        self.max_route_steps = 20        # how many expansions in build_route

        # to store how many precomputed solutions we'll generate
        self.num_precomputed_solutions = 3

class RRTACOOptimizer:
    def __init__(self, G, config, station_nodes=None):
        self.G = G
        self.config = config
        self.station_nodes = station_nodes.copy() if station_nodes else []
        self.current_station_index = 0  # Track the current index for round-robin selection


        self.edge_length_cache = {}
        self.adjacency_cache = {}

        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 0)
            self.edge_length_cache[(u, v)] = length

        for node in self.G.nodes():
            # store successors
            self.adjacency_cache[node] = list(self.G.successors(node))

        # all edges for coverage
        self.all_edges = set(self.edge_length_cache.keys())

        # Initialize pheromone
        self.pheromone = {}
        # start uniform
        for (u, v) in self.edge_length_cache.keys():
            self.pheromone[(u, v)] = 1.0

        self.best_solution = None
        self.best_fitness = -float('inf')
        

    def build_ant_route(self):
        """
        Build a single route (for one ant = one vehicle) incorporating ACO-style edge choice.
        Starts from police station nodes in a round-robin fashion if available.
        """
        nodes_list = list(self.G.nodes())
        if not nodes_list:
            return []

        # Choose starting node: police station (if available) in round-robin, else random
        if self.station_nodes:
            current_idx = self.current_station_index % len(self.station_nodes)
            current = self.station_nodes[current_idx]
            self.current_station_index += 1  # Move to next station for the next ant
        else:
            current = random.choice(nodes_list)

        route = [current]
        dist_so_far = 0.0

        for _ in range(self.config.max_route_steps):
            # build probability distribution over all nodes
            # Weighted by "sum of pheromone" in traveling from current -> node 
            # times 1/distance. We'll do a simplified approach
            chosen_node = self.select_next_node_global(current)

            if chosen_node is None or chosen_node == current:
                continue

            # shortest path from current -> chosen_node
            try:
                sp = nx.shortest_path(self.G, current, chosen_node, weight='length')
            except nx.NetworkXNoPath:
                continue

            sub_len = 0.0
            for u, v in zip(sp[:-1], sp[1:]):
                if (u, v) in self.edge_length_cache:
                    sub_len += self.edge_length_cache[(u, v)]
                elif (v, u) in self.edge_length_cache:
                    sub_len += self.edge_length_cache[(v, u)] 

            if dist_so_far + sub_len > self.config.max_route_distance:
                break

            # append path
            for idx in range(1, len(sp)):
                route.append(sp[idx])
            dist_so_far += sub_len
            current = chosen_node

        return route

    def select_next_node_global(self, current):
        """
        We pick among all possible nodes in G using a 
        probability distribution ~ sum of pheromone edges 
        on the shortest path from current -> node.
        This can get expensive for large graphs, so we keep it 
        simpler: just pick random node for demonstration.
        """
        nodes_list = list(self.G.nodes())
        if not nodes_list:
            return None
        # In a real ACO you'd do something more sophisticated,
        # but let's just pick random to keep it simpler
        return random.choice(nodes_list)

    def build_ant_solution(self):
        """
        A solution = num_ants routes, one route per ant.
        """
        return [self.build_ant_route() for _ in range(self.config.num_ants)]

    def fitness(self, solution):
        """
        coverage reward: coverage_factor * sum(edge_length for covered edges)
        overlap penalty if more than 3 vehicles on same edge
        """
        edge_usage = {}
        covered_edges = set()

        for route in tqdm(solution,desc="Computing Fitness"):
            for u, v in zip(route[:-1], route[1:]):
                if (u, v) in self.edge_length_cache:
                    edge_usage[(u, v)] = edge_usage.get((u, v), 0) + 1
                    covered_edges.add((u, v))
                elif (v, u) in self.edge_length_cache:
                    edge_usage[(v, u)] = edge_usage.get((v, u), 0) + 1
                    covered_edges.add((v, u))

        coverage_reward = 0
        for e in covered_edges:
            coverage_reward += self.config.coverage_factor * self.edge_length_cache[e]

        overlap_penalty = 0
        for e, count in edge_usage.items():
            if count > 3:
                overlap_penalty += self.config.penalty_factor * (count - 3) * self.edge_length_cache[e]

        return coverage_reward - overlap_penalty

    def seed_pheromones(self, rrt_solutions, seed_amount=2.0):
        """
        For each route in each precomputed rrt_solutions,
        deposit extra pheromone on the edges used.
        This helps 'guide' the ants to explore those edges early on.
        """
        for sol in tqdm(rrt_solutions,desc="Seeding Pheromones"):
            for route in sol:
                for u, v in zip(route[:-1], route[1:]):
                    if (u, v) in self.pheromone:
                        self.pheromone[(u, v)] += seed_amount
                    elif (v, u) in self.pheromone:
                        self.pheromone[(v, u)] += seed_amount

    def run(self, precomputed_solutions=[]):
        fitness_progress = []

        for iteration in range(self.config.num_iterations):
            solution = self.build_ant_solution()
            fit = self.fitness(solution)
            fitness_progress.append(fit)
            
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_solution = solution

            self.evaporate_pheromones()
            self.deposit_pheromones(solution, fit)

        return self.best_solution, self.best_fitness, fitness_progress

    def evaporate_pheromones(self):
        rho = self.config.evaporation_rate
        for e in self.pheromone:
            self.pheromone[e] = (1.0 - rho) * self.pheromone[e]
            if self.pheromone[e] < 0:
                self.pheromone[e] = 0

    def deposit_pheromones(self, solution, fit_value):
        deposit_amount = max(fit_value, 1)
        for route in solution:
            for u, v in zip(route[:-1], route[1:]):
                if (u, v) in self.pheromone:
                    self.pheromone[(u, v)] += (self.config.Q * deposit_amount / 1000.0)
                elif (v, u) in self.pheromone:
                    self.pheromone[(v, u)] += (self.config.Q * deposit_amount / 1000.0)

###############################################################################
# 5) CONVERT SOLUTION TO GEOJSON
###############################################################################
def solution_to_geojson(G, solution):
    """
    Convert a solution (list of routes) into a GeoJSON FeatureCollection.
    Each route gets a unique color.
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
            if (u, v) in G.edges():
                edge_data = G.get_edge_data(u, v)
                if edge_data:
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
    st.title("ACO Patrol Route Optimization with Police Station Starting Points")

    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles (Ants)", 2, 50, 50)
        num_iterations = st.slider("ACO Iterations", 10, 200, 50)
        coverage_factor = st.slider("Coverage Factor", 1000, 20000, 5000, step=1000)
        overlap_penalty_factor = st.slider("Overlap Penalty Factor", 1000, 50000, 10000, step=1000)
        max_route_distance = st.slider("Max Route Distance", 5000, 50000, 15000, step=5000)
        max_route_steps = st.slider("Max Route Steps", 5, 50, 20)
        num_precomputed = st.slider("Num RRT Solutions to Generate", 1, 10, 3)
        run_btn = st.button("Run Optimization")

    if run_btn:
        start_time = time.time()
        
        # 1) Initialize configuration
        config = ACOConfig()
        config.num_ants = num_vehicles
        config.num_iterations = num_iterations
        config.coverage_factor = coverage_factor
        config.penalty_factor = overlap_penalty_factor
        config.max_route_distance = max_route_distance
        config.max_route_steps = max_route_steps
        config.num_precomputed_solutions = num_precomputed

        # 2) Load road network
        with st.spinner("Loading Coimbatore road network..."):
            G = load_road_network("Coimbatore, India")

        # 3) Get police stations
        with st.spinner("Identifying police stations..."):
            station_nodes, police_gdf = get_police_station_nodes(G, "Coimbatore, India")
            if police_gdf is not None:
                st.success(f"Found {len(police_gdf)} police stations")
            else:
                st.warning("No police stations found - using random starting points")

        # 4) Generate RRT solutions
        with st.spinner("Generating initial solutions with RRT..."):
            precomputed_solutions = generate_precomputed_solutions(
                G=G,
                how_many=config.num_precomputed_solutions,
                num_vehicles=config.num_ants,
                max_route_distance=config.max_route_distance,
                max_route_steps=config.max_route_steps
            )

        # 5) Run ACO optimization
        with st.spinner("Optimizing patrol routes..."):
            optimizer = RRTACOOptimizer(G, config, station_nodes)
            best_sol, best_fit, fitness_progress = optimizer.run(precomputed_solutions)

        # 6) Visualize results
        st.subheader("Optimization Results")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create Leafmap visualization
            m = leafmap.Map(center=[11.0168, 76.9558], zoom=12, height=600)
            
            # Add subtle boundary
            boundary_gdf = ox.geocode_to_gdf("Coimbatore, India")
            m.add_gdf(
                boundary_gdf,
                layer_name="City Boundary",
                style={
                    "color": "#444444",
                    "fillOpacity": 0.05,
                    "weight": 1,
                    "dashArray": "2,2"
                }
            )

            # Add police stations
            if police_gdf is not None:
                police_geojson = ox.io._geometries_to_geojson(police_gdf)
                m.add_geojson(
                    police_geojson,
                    layer_name="Police Stations",
                    style={
                        "color": "#0044FF",
                        "markerSize": 10,
                        "markerSymbol": "police",
                        "fillOpacity": 0.8
                    }
                )

            # Add optimized routes
            route_geojson = solution_to_geojson(G, best_sol)
            m.add_geojson(
                route_geojson,
                layer_name="Patrol Routes",
                style={"stroke-width": 2}
            )

            # Add layer control
            m.add_layer_control(position="topright")
            
            # Display map
            st.write("### Patrol Route Visualization")
            m.to_streamlit()

        with col2:
            # Show fitness progress
            st.write("### Optimization Progress")
            plt.figure(figsize=(8, 4))
            plt.plot(fitness_progress, color="#00aa00")
            plt.xlabel("Iteration")
            plt.ylabel("Fitness Score")
            plt.grid(alpha=0.3)
            st.pyplot(plt)

            # Display metrics
            st.write("### Key Metrics")
            st.metric("Best Fitness Score", f"{best_fit:,.2f}")
            st.metric("Total Route Distance", 
                     f"{(sum(len(r) for r in best_sol)/1000):.1f} km")
            st.metric("Computation Time", 
                     f"{(time.time() - start_time):.2f} seconds")

if __name__ == "__main__":
    main()
    
    
