import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
import random
import numpy as np
from time import sleep
from osmnx.distance import nearest_nodes


# Configure OSMnx caching and logging
ox.settings.use_cache = True
ox.settings.log_console = True

def load_road_network(place="Coimbatore, India", cache_dir="road_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    safe_place = place.replace(" ", "_").replace(",", "")
    cache_file = os.path.join(cache_dir, f"{safe_place}_network.graphml")
    if os.path.exists(cache_file):
        st.write(f"Loading cached road network from {cache_file}")
        G = ox.load_graphml(cache_file)
    else:
        st.write(f"Downloading road network for {place}")
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        ox.save_graphml(G, cache_file)
    return G

def get_major_edges(G):
    """Return a set of edges considered 'major' based on their highway attribute."""
    major_types = {"motorway", "trunk", "primary", "secondary"}
    major_edges = set()
    for u, v, data in G.edges(data=True):
        highway = data.get('highway')
        if highway:
            if isinstance(highway, list):
                if any(h in major_types for h in highway):
                    major_edges.add((u, v))
            else:
                if highway in major_types:
                    major_edges.add((u, v))
    return major_edges

def get_major_nodes(G):
    """Extract major nodes from major edges."""
    major_edges = get_major_edges(G)
    major_nodes = set()
    for u, v in major_edges:
        major_nodes.add(u)
        major_nodes.add(v)
    return major_nodes

def police_station_start_nodes(G, police_coords):
    start_nodes = []
    for coord in police_coords:
        node = nearest_nodes(G, coord[1], coord[0])  # switched from get_nearest_node
        start_nodes.append(node)
    return start_nodes
class GAConfig:
    def __init__(self):
        self.population_size = 50
        self.num_generations = 100
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        self.num_vehicles = 5
        self.max_route_distance = 15000  # in meters
        self.penalty_factor = 10000
        self.major_node_penalty = 50000  # penalty for each major node not covered

class RouteOptimizerGA:
    def __init__(self, G, config, start_nodes, progress_callback=None):
        self.G = G
        self.config = config
        self.start_nodes = start_nodes  # fixed starting node per vehicle
        self.num_vehicles = config.num_vehicles
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.major_nodes = get_major_nodes(G)
        self.progress_callback = progress_callback

    def random_route(self, start):
        # Generate a random route that starts at 'start'
        route, total_dist = [start], 0
        current = start
        while total_dist < self.config.max_route_distance:
            neighbors = list(self.G.successors(current))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            edge_data = self.G.get_edge_data(current, next_node)
            length = edge_data[0].get('length', 0) if edge_data else 0
            if total_dist + length > self.config.max_route_distance:
                break
            route.append(next_node)
            total_dist += length
            current = next_node
        # Force Hamiltonian cycle: return to start if possible
        if route[-1] != start:
            try:
                cycle = nx.shortest_path(self.G, route[-1], start, weight='length')
                route.extend(cycle[1:])
                total_dist += sum(self.G.get_edge_data(u, v)[0].get('length', 0) for u, v in zip(cycle[:-1], cycle[1:]))
            except nx.NetworkXNoPath:
                # If no path, add heavy penalty later in fitness
                pass
        return route

    def initialize_population(self):
        # Each individual is a list of routes, one per vehicle. Each route must start (and end) at its designated start node.
        population = []
        for _ in range(self.config.population_size):
            individual = []
            for i in range(self.num_vehicles):
                route = self.random_route(self.start_nodes[i])
                individual.append(route)
            population.append(individual)
        return population

    def fitness(self, individual):
        route_lengths, edge_usage, node_usage = [], {}, {}
        for route in individual:
            # Ensure route is a cycle: if not, add penalty for missing connection
            start = route[0]
            if route[-1] != start:
                try:
                    cycle = nx.shortest_path(self.G, route[-1], start, weight='length')
                    route = route + cycle[1:]
                except nx.NetworkXNoPath:
                    return -float('inf')  # invalid route
            route_length = 0
            for u, v in zip(route[:-1], route[1:]):
                edge_data = self.G.get_edge_data(u, v)
                length = edge_data[0].get('length', 0) if edge_data else 0
                route_length += length
                edge_usage[(u, v)] = edge_usage.get((u, v), 0) + 1
                node_usage[u] = node_usage.get(u, 0) + 1
                node_usage[v] = node_usage.get(v, 0) + 1
            route_lengths.append(route_length)
        total_length = sum(route_lengths)
        # Constraint: at most 3 vehicles per edge
        edge_penalty = sum(
            self.config.penalty_factor * (count - 3) * self.G.get_edge_data(edge[0], edge[1])[0].get('length', 0)
            for edge, count in edge_usage.items() if count > 3
        )
        # Constraint: equal route lengths
        variance_penalty = np.var(route_lengths) * 100
        # Constraint: cover all major nodes (each major node must appear in at least one route)
        covered_major_nodes = set()
        for route in individual:
            covered_major_nodes.update(route)
        missing_major = self.major_nodes - covered_major_nodes
        major_node_penalty = self.config.major_node_penalty * len(missing_major)
        fitness_value = total_length - edge_penalty - variance_penalty - major_node_penalty
        return fitness_value

    def selection(self, population):
        return [max(random.sample(population, 2), key=self.fitness)
                for _ in range(len(population))]

    def crossover(self, parent1, parent2):
        child = []
        for i in range(self.num_vehicles):
            route1 = parent1[i]
            route2 = parent2[i]
            start = self.start_nodes[i]
            if len(route1) < 2 or len(route2) < 2 or random.random() > self.config.crossover_rate:
                child.append(route1.copy())
            else:
                cut1 = random.randint(1, len(route1) - 1)
                cut2 = random.randint(1, len(route2) - 1)
                new_route = route1[:cut1] + route2[cut2:]
                # Repair route to ensure continuity
                repaired_route = [new_route[0]]
                for j in range(1, len(new_route)):
                    prev, curr = repaired_route[-1], new_route[j]
                    if not self.G.has_edge(prev, curr):
                        try:
                            sp = nx.shortest_path(self.G, prev, curr, weight='length')
                            repaired_route.extend(sp[1:])
                        except nx.NetworkXNoPath:
                            repaired_route.append(curr)
                    else:
                        repaired_route.append(curr)
                # Force cycle: ensure route ends at starting node
                if repaired_route[-1] != start:
                    try:
                        sp = nx.shortest_path(self.G, repaired_route[-1], start, weight='length')
                        repaired_route.extend(sp[1:])
                    except nx.NetworkXNoPath:
                        pass
                child.append(repaired_route)
        return child

    def mutate(self, individual):
        # Mutate one route in the individual
        ind = [r.copy() for r in individual]
        vehicle_idx = random.randint(0, self.num_vehicles - 1)
        route = ind[vehicle_idx]
        if len(route) < 4:
            return ind
        i = random.randint(0, len(route) - 3)
        j = random.randint(i + 2, len(route) - 1)
        try:
            sp = nx.shortest_path(self.G, route[i], route[j], weight='length')
            ind[vehicle_idx] = route[:i+1] + sp[1:] + route[j+1:]
        except nx.NetworkXNoPath:
            pass
        # Ensure mutated route still starts and ends at the same police station node
        start = self.start_nodes[vehicle_idx]
        if ind[vehicle_idx][0] != start:
            ind[vehicle_idx][0] = start
        if ind[vehicle_idx][-1] != start:
            try:
                sp = nx.shortest_path(self.G, ind[vehicle_idx][-1], start, weight='length')
                ind[vehicle_idx].extend(sp[1:])
            except nx.NetworkXNoPath:
                pass
        return ind

    def run(self):
        population = self.initialize_population()
        best, best_fit = None, -float('inf')
        for gen in range(self.config.num_generations):
            for ind in population:
                fit = self.fitness(ind)
                if fit > best_fit:
                    best_fit = fit
                    best = ind
            if self.progress_callback:
                self.progress_callback(gen + 1, self.config.num_generations)
            selected = self.selection(population)
            new_population = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[i+1] if i+1 < len(selected) else selected[0]
                c1 = self.crossover(p1, p2)
                c2 = self.crossover(p2, p1)
                if random.random() < self.config.mutation_rate:
                    c1 = self.mutate(c1)
                if random.random() < self.config.mutation_rate:
                    c2 = self.mutate(c2)
                new_population.extend([c1, c2])
            population = new_population[:self.config.population_size]
        self.best_solution, self.best_fitness = best, best_fit
        return best, best_fit

def solution_to_geojson(G, solution):
    # Store each vehicle's route as a separate GeoJSON layer
    layers = {}
    for vid, route in enumerate(solution):
        if len(route) < 2:
            continue
        coords = []
        for u, v in zip(route[:-1], route[1:]):
            try:
                line = list(LineString([(G.nodes[u]['x'], G.nodes[u]['y']),
                                          (G.nodes[v]['x'], G.nodes[v]['y'])]).coords)
                coords.append(line)
            except Exception:
                continue
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"vehicle": vid + 1},
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": coords
                }
            }]
        }
        layers[f"Police Station {vid+1}"] = geojson
    return layers

def main():
    st.set_page_config(layout="wide")
    st.title("🚔 City-Wide Vehicle Coverage (GA Optimized)")

    # Use session_state to cache the solution so that subsequent parameter changes do not auto-run optimization
    if 'ga_solution' not in st.session_state:
        st.session_state.ga_solution = None
        st.session_state.best_fitness = None

    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles", 3, 10, 5)
        # Set police station coordinates (latitude, longitude) for Coimbatore (dummy values)
        police_coords = [
            (11.0168, 76.9558),
            (11.0300, 76.9700),
            (11.0100, 76.9400),
            (11.0250, 76.9600),
            (11.0050, 76.9500),
            (11.0200, 76.9650),
            (11.0150, 76.9600),
            (11.0080, 76.9550),
            (11.0120, 76.9500),
            (11.0180, 76.9700)
        ]
        # Use only as many police station coordinates as vehicles
        police_coords = police_coords[:num_vehicles]

        run_algo = st.button("Run Optimization")
        reset_algo = st.button("Reset Optimization")

    # Load road network
    G = load_road_network("Coimbatore, India")
    # Get starting nodes for police stations (consistent across runs)
    start_nodes = police_station_start_nodes(G, police_coords)

    config = GAConfig()
    config.num_vehicles = num_vehicles

    # Progress callback using Streamlit progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    def progress_callback(current, total):
        progress_bar.progress(current / total)
        status_text.text(f"Generation {current} of {total}")
        sleep(0.01)

    # If reset, clear cached solution
    if reset_algo:
        st.session_state.ga_solution = None
        st.session_state.best_fitness = None

    # Run optimization if not already computed or if run button pressed
    if run_algo or st.session_state.ga_solution is None:
        optimizer = RouteOptimizerGA(G, config, start_nodes, progress_callback=progress_callback)
        best_solution, best_fit = optimizer.run()
        st.session_state.ga_solution = best_solution
        st.session_state.best_fitness = best_fit

    st.sidebar.write(f"Best Fitness: {st.session_state.best_fitness:.1f}")

    # Prepare separate layers for each vehicle route
    layers = solution_to_geojson(G, st.session_state.ga_solution)
    m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
    for layer_name, geojson_data in layers.items():
        # Add a checkbox for each layer
        if st.sidebar.checkbox(layer_name, value=True):
            m.add_geojson(geojson_data, layer_name=layer_name)

    # Show map in a dedicated container
    st.subheader("Optimized Routes")
    m.to_streamlit(height=600)

if __name__ == "__main__":
    main()
