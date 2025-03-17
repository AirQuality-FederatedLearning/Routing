import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import random
import numpy as np
from shapely.geometry import LineString
from tqdm import tqdm


# --------------------------------------------------------------------
# 1) Load/Cache Road Network
# --------------------------------------------------------------------
def load_road_network(place="Coimbatore, India", cache_dir="road_cache"):
    """
    Loads or downloads a road network for the specified place.
    Uses a local GraphML cache to avoid repeated downloads.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_place = place.replace(" ", "_").replace(",", "")
    cache_file = os.path.join(cache_dir, f"{safe_place}_network.graphml")

    # Example path to major roads if you have it locally
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

# --------------------------------------------------------------------
# 2) PSO + RRT* Config
# --------------------------------------------------------------------
class PSOConfig:
    
    
    def __init__(self):
        self.num_vehicles = 5          # Vehicles
        self.swarm_size = 20          # Number of particles
        self.num_iterations = 50

        # PSO parameters
        self.w = 0.5                  # Inertia weight
        self.c1 = 1.5                 # Cognitive factor
        self.c2 = 1.5                 # Social factor

        # Coverage / overlap parameters
        self.coverage_factor = 5000   # Reward factor for covering new edges
        self.overlap_penalty_factor = 10000 # Penalty factor if more than 3 vehicles use an edge

        # RRT-like expansions
        self.max_route_steps = 50     # Max number of steps in a route
        self.explore_prob = 0.3       # Probability to expand to an unvisited neighbor if possible

# --------------------------------------------------------------------
# 3) PSO + RRT* Optimizer
# --------------------------------------------------------------------
class PSORRTOptimizer:
    def __init__(self, G, config):
        self.G = G
        self.config = config

        # Build adjacency + edge length cache for fast lookups
        self.adjacency_cache = {}
        self.edge_length_cache = {}
        for u in self.G.nodes():
            self.adjacency_cache[u] = list(self.G.successors(u))
        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 0)
            self.edge_length_cache[(u, v)] = length

        # Build list of all directed edges for coverage tracking
        self.all_edges = set(self.edge_length_cache.keys())

        # Initialize swarm
        self.swarm = []
        self.personal_best = []
        self.personal_best_fitness = []

        self.global_best = None
        self.global_best_fitness = -float('inf')

    # --------------- RRT-Like Random Route Generation ---------------
    def build_rrt_route(self):
        """
        Build a single route using an RRT*-inspired approach:
        - Start from a random node
        - At each step, randomly pick one successor
          with some bias toward "less visited" or "unexplored" edges
        """
        nodes_list = list(self.G.nodes())
        if not nodes_list:
            return []
        current = random.choice(nodes_list)
        route = [current]

        for _ in range(self.config.max_route_steps):
            neighbors = self.adjacency_cache[current]
            if not neighbors:
                break
            # Explore vs. exploit
            if random.random() < self.config.explore_prob:
                # Try to pick a neighbor with fewer edges used or an unvisited neighbor
                random.shuffle(neighbors)
            next_node = random.choice(neighbors)
            route.append(next_node)
            current = next_node
        return route

    def build_solution(self):
        """
        Build a solution = list of routes, one route per vehicle.
        """
        return [self.build_rrt_route() for _ in range(self.config.num_vehicles)]

    # --------------- Fitness ---------------
    def fitness(self, solution):
        """
        Coverage-based fitness:
          1) coverage_reward = coverage_factor * sum(length of newly covered edges)
          2) overlap penalty if more than 3 vehicles use the same edge
          final = coverage_reward - overlap_penalty
        """
        covered_edges = set()
        edge_usage = {}

        for route in solution:
            # Walk the route, edge by edge
            for u, v in zip(route[:-1], route[1:]):
                if (u, v) in self.edge_length_cache:
                    edge_usage[(u, v)] = edge_usage.get((u, v), 0) + 1
                    covered_edges.add((u, v))
                # Also check reversed edge if your G has them
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

    # --------------- Particle Initialization ---------------
    def initialize_swarm(self):
        self.swarm = []
        self.personal_best = []
        self.personal_best_fitness = []

        for _ in range(self.config.swarm_size):
            sol = self.build_solution()
            fit = self.fitness(sol)
            self.swarm.append(sol)
            self.personal_best.append(sol)
            self.personal_best_fitness.append(fit)

            if fit > self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best = sol

    # --------------- Represent "Velocity" for route-based solutions ---------------
    def velocity_update(self, i, sol):
        """
        For route-based solutions, we define "velocity" in a purely symbolic way:
          - 'Pull' the current solution a bit toward personal_best[i] and global_best
          - We do that by picking some fraction of routes from personal_best and global_best
            and substituting them or merging them into the current solution.

        This is not a classic numeric velocity, but a meta-heuristic approach typical for
        combinatorial PSO variants.
        """
        pbest = self.personal_best[i]
        gbest = self.global_best

        new_sol = []
        for route_idx in range(self.config.num_vehicles):
            old_route = sol[route_idx]
            r = random.random()
            if r < 0.33:
                # keep old route
                new_sol.append(old_route)
            elif r < 0.66:
                # copy route from personal best
                new_sol.append(pbest[route_idx])
            else:
                # copy route from global best
                new_sol.append(gbest[route_idx])

        # We can do a partial crossover between old route and best routes, etc.
        # Let's keep it simple: we do direct substitution for demonstration.
        return new_sol

    # --------------- Local "Mutation" or RRT Re-expansion ---------------
    def mutate(self, sol):
        """
        For each route, with small probability, do an RRT re-expansion in the middle of the route.
        """
        new_sol = []
        for route in sol:
            if random.random() < 0.2:
                # pick a random index
                if len(route) > 2:
                    i = random.randint(1, len(route)-2)
                    # from that node, do a small new random expansion
                    expansion = self.build_rrt_route()
                    # merge them
                    new_route = route[:i] + expansion + route[i:]
                    new_sol.append(new_route)
                else:
                    new_sol.append(route)
            else:
                new_sol.append(route)
        return new_sol

    # --------------- Main PSO Loop ---------------
    def run(self):
        self.initialize_swarm()

        for it in range(self.config.num_iterations):
            for i in range(self.config.swarm_size):
                # "Velocity" update
                updated = self.velocity_update(i, self.swarm[i])
                # Mutation / RRT expansions
                mutated = self.mutate(updated)

                fit = self.fitness(mutated)
                # Accept if improved
                if fit > self.fitness(self.swarm[i]):
                    self.swarm[i] = mutated
                    # Update personal best
                    if fit > self.personal_best_fitness[i]:
                        self.personal_best[i] = mutated
                        self.personal_best_fitness[i] = fit
                        # Update global best
                        if fit > self.global_best_fitness:
                            self.global_best = mutated
                            self.global_best_fitness = fit

        return self.global_best, self.global_best_fitness

# --------------------------------------------------------------------
# 4) Convert Solution to GeoJSON for Leafmap
# --------------------------------------------------------------------
def solution_to_geojson(G, solution):
    """
    solution: list of routes (each route is a list of nodes)
    We'll build MultiLineString geometry for each route, color-coded.
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
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                if edge_data is not None:
                    # Some edges have multiple keys
                    edge_dict = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
                    geom = edge_dict.get("geometry", None)
                    if geom is None:
                        geom = LineString([
                            (G.nodes[u]["x"], G.nodes[u]["y"]),
                            (G.nodes[v]["x"], G.nodes[v]["y"]),
                        ])
                    lines.append(geom)
            elif G.has_edge(v, u):
                edge_data = G.get_edge_data(v, u)
                if edge_data is not None:
                    edge_dict = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
                    geom = edge_dict.get("geometry", None)
                    if geom is None:
                        geom = LineString([
                            (G.nodes[v]["x"], G.nodes[v]["y"]),
                            (G.nodes[u]["x"], G.nodes[u]["y"]),
                        ])
                    lines.append(geom)
            # else skip if there's no direct edge

        if lines:
            color = colors[vid % len(colors)]
            feature = {
                "type": "Feature",
                "properties": {
                    "vehicle": vid+1,
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

# --------------------------------------------------------------------
# 5) Streamlit Main
# --------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("PSO + RRT* Coverage Demo")

    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles", 2, 50, 50)
        swarm_size = st.slider("Swarm Size", 5, 50, 20)
        num_iterations = st.slider("Iterations", 10, 200, 50)

        coverage_factor = st.slider("Coverage Factor", 1000, 20000, 5000, step=1000,
                                    help="Reward factor for covering new edges.")
        overlap_penalty_factor = st.slider("Overlap Penalty Factor", 1000, 50000, 10000, step=1000,
                                           help="Penalty factor for edges used by >3 vehicles.")
        run_btn = st.button("Run PSO Coverage")

    if run_btn:
        config = PSOConfig()
        config.num_vehicles = num_vehicles
        config.swarm_size = swarm_size
        config.num_iterations = num_iterations
        config.coverage_factor = coverage_factor
        config.overlap_penalty_factor = overlap_penalty_factor

        with st.spinner("Loading road network..."):
            G = load_road_network("Coimbatore, India")

        with st.spinner("Running PSO + RRT expansions..."):
            optimizer = PSORRTOptimizer(G, config)
            best_solution, best_fit = optimizer.run()

        # Convert best solution to GeoJSON
        geojson_data = solution_to_geojson(G, best_solution)

        # Display in Leafmap
        m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
        # If you want boundary for Coimbatore:
        boundary_gdf = ox.geocode_to_gdf("Coimbatore, India")
        m.add_gdf(boundary_gdf, layer_name="Coimbatore Boundary",
                  style={"color": "#FF0000", "fillOpacity": 0.1})
        m.add_geojson(geojson_data, layer_name="PSO-Routes")

        st.success(f"Best Fitness Achieved: {best_fit:.2f}")
        m.to_streamlit()

if __name__ == "__main__":
    main()
