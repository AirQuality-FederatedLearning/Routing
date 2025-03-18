import os
import copy
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import osmnx as ox
import networkx as nx
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
    # If you have a local "major roads" file, adjust this path:
    major_road_network = r"C:\Users\DELL\Documents\Amrita\4th year\ArcGis\major_road_cache\Coimbatore_India_major.graphml"
    if os.path.exists(major_road_network):
        print(f"Loading cached road network from {major_road_network}")
        G = ox.load_graphml(major_road_network)
    elif os.path.exists(cache_file):
        print(f"Loading cached road network from {cache_file}")
        G = ox.load_graphml(cache_file)
    else:
        print(f"Downloading road network for {place}")
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        ox.save_graphml(G, cache_file)
    return G

###############################################################################
# 2) GET POLICE STATION NODES (Fixed Starting Points)
###############################################################################
def get_police_station_nodes(G, place="Coimbatore, India"):
    """
    1) Fetch police stations from OSM for the given place using the amenity tag.
    2) For each police station, find the nearest node in the road network G.
    3) Return a list of valid node IDs.
    """
    try:
        gdf_police = ox.features_from_place(place, tags={"amenity": "police"})
    except Exception as e:
        print(f"Could not fetch police stations: {e}")
        return []
    if gdf_police.empty:
        print("No police stations found; using random nodes.")
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
# 3) PSO CONFIGURATION (Adjusted Scaling for Easy Comparison)
###############################################################################
class PSOConfig:
    def __init__(self):
        self.num_vehicles = 5          # Number of routes (vehicles) per solution
        self.swarm_size = 30           # Number of particles
        self.num_iterations = 50

        # PSO parameters (conceptual in this discrete setting)
        self.w = 0.5   # inertia (not heavily used)
        self.c1 = 1.5  # cognitive factor
        self.c2 = 1.5  # social factor

        # Scaled reward/penalty factors (lower for easier comparison)
        self.coverage_factor = 1      # Reward per meter covered
        self.overlap_penalty_factor = 1  # Penalty per meter overlapped

        self.max_route_distance = 15000  # Maximum allowed route length (meters)
        self.mutation_rate = 0.3         # Increased mutation for exploration

###############################################################################
# 4) PSO ROUTE OPTIMIZER (Using Police Stations as Fixed Starting Points)
###############################################################################
class PSORouteOptimizer:
    def __init__(self, G, config, police_nodes=None):
        self.G = G
        self.config = config
        self.police_nodes = police_nodes if police_nodes else []
        
        # Build adjacency and edge-length caches
        self.adjacency_cache = {node: list(self.G.successors(node)) for node in self.G.nodes()}
        self.edge_length_cache = {}
        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 0)
            self.edge_length_cache[(u, v)] = length
        self.all_edges = set(self.edge_length_cache.keys())
        
        # Swarm data structures
        self.swarm = []               # List of solutions (each is a list of routes)
        self.personal_best = []
        self.personal_best_fitness = []
        self.global_best = None
        self.global_best_fitness = -float('inf')
        self.frames = []              # To store frames for animation

    def build_route_from_start(self, start):
        """Build a random route that always begins with the fixed start node."""
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

    def build_solution(self):
        """A solution consists of one route per vehicle. Each route starts at a fixed police station if available."""
        solution = []
        for i in range(self.config.num_vehicles):
            if self.police_nodes:
                start = random.choice(self.police_nodes)
            else:
                start = random.choice(list(self.G.nodes()))
            route = self.build_route_from_start(start)
            solution.append(route)
        return solution

    def fitness(self, solution):
        """Compute fitness as coverage reward minus overlap penalty."""
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
        overlap_penalty = 0
        for e, count in edge_usage.items():
            if count > 3:
                overlap_penalty += self.config.overlap_penalty_factor * (count - 3) * self.edge_length_cache[e]
        return coverage_reward - overlap_penalty

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
                self.global_best = sol
                self.global_best_fitness = fit

    def velocity_update(self, i, current_sol):
        """
        For each vehicle route, choose from current, personal best, or global best.
        Ensure the starting node (police station) remains unchanged.
        """
        pbest = self.personal_best[i]
        gbest = self.global_best
        new_sol = []
        for v in range(self.config.num_vehicles):
            start = current_sol[v][0]  # fixed starting point
            r = random.random()
            if r < 0.34:
                new_route = current_sol[v]
            elif r < 0.67:
                new_route = pbest[v]
            else:
                new_route = gbest[v]
            new_route[0] = start  # enforce fixed starting point
            new_sol.append(new_route)
        return new_sol

    def mutate(self, solution):
        """With a probability, mutate a segment of each route while preserving the starting node."""
        new_sol = []
        for route in solution:
            start = route[0]
            if random.random() < self.config.mutation_rate and len(route) > 4:
                i = random.randint(1, len(route) - 2)  # start index 1 onward
                j = random.randint(i + 1, len(route) - 1)
                sub_start = route[i]
                sub_end = route[j]
                sub_route = self._build_subroute(sub_start, sub_end)
                mutated_route = route[:i] + sub_route[1:-1] + route[j:]
                mutated_route[0] = start
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

    def run(self):
        """Main PSO loop that updates swarm, personal bests, and global best, storing frames for animation."""
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
            self.frames.append((iteration, copy.deepcopy(self.global_best), self.global_best_fitness))
        return self.global_best, self.global_best_fitness, self.frames

###############################################################################
# CONVERT SOLUTION TO LINES FOR PLOTTING
###############################################################################
def solution_to_lines(G, solution):
    segments = []
    for route in solution:
        seg = []
        for node in route:
            x = G.nodes[node]['x']
            y = G.nodes[node]['y']
            seg.append((x, y))
        segments.append(seg)
    return segments

###############################################################################
# GET BACKGROUND BOUNDARY COORDINATES
###############################################################################
def get_boundary_coords(place="Coimbatore, India"):
    try:
        boundary_gdf = ox.geocode_to_gdf(place)
        geom = boundary_gdf.geometry.iloc[0]
        if geom.geom_type == "Polygon":
            return list(geom.exterior.coords)
        elif geom.geom_type == "MultiPolygon":
            largest = max(geom.geoms, key=lambda p: p.area)
            return list(largest.exterior.coords)
    except Exception as e:
        print(f"Error fetching boundary: {e}")
    return None

###############################################################################
# UPDATE FUNCTION FOR MATPLOTLIB ANIMATION
###############################################################################
def update(frame, G, ax, boundary_coords):
    iteration, solution, fitness = frame
    ax.clear()
    ax.set_title(f"Iteration: {iteration} | Fitness: {fitness:.2f}")
    # Plot background boundary
    if boundary_coords:
        bx, by = zip(*boundary_coords)
        ax.plot(bx, by, color="gray", linewidth=1, linestyle="--")
    segments = solution_to_lines(G, solution)
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "olive", "cyan", "magenta"]
    for i, seg in enumerate(segments):
        if len(seg) < 2:
            continue
        xs, ys = zip(*seg)
        ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=2, marker='o', label=f'Route {i+1}')
    ax.legend(loc="upper right")
    ax.set_aspect('equal')
    # Set axis limits based on graph nodes
    all_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()]
    xs_all, ys_all = zip(*all_coords)
    margin = 0.001
    ax.set_xlim(min(xs_all)-margin, max(xs_all)+margin)
    ax.set_ylim(min(ys_all)-margin, max(ys_all)+margin)

###############################################################################
# MAIN FUNCTION (ANIMATION)
###############################################################################
def main():
    print("Loading road network...")
    G = load_road_network("Coimbatore, India")
    print("Fetching police station nodes...")
    police_nodes = get_police_station_nodes(G, "Coimbatore, India")
    print(f"Found {len(police_nodes)} police station nodes.")
    
    config = PSOConfig()
    optimizer = PSORouteOptimizer(G, config, police_nodes)
    best_solution, best_fit, frames = optimizer.run()
    print(f"Optimization complete. Best Fitness: {best_fit:.2f}")
    
    boundary_coords = get_boundary_coords("Coimbatore, India")
    fig, ax = plt.subplots(figsize=(8,8))
    anim = FuncAnimation(fig, update, frames=frames, fargs=(G, ax, boundary_coords), interval=500)
    plt.show()
    # Optionally, save the animation:
    anim.save("pso_animation.mp4", writer="ffmpeg")
    # or
    anim.save("pso_animation.gif", writer="imagemagick")

if __name__ == "__main__":
    main()
