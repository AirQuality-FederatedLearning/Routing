import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm  # Import tqdm for progress visualization

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

class GAConfig:
    def __init__(self):
        self.population_size = 50
        self.num_generations = 100
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        self.num_vehicles = 50
        self.max_route_distance = 15000  # Maximum allowed route distance (meters)
        self.penalty_factor = 10000      # Penalty factor for each extra vehicle using an edge

class RouteOptimizerGA:
    def __init__(self, G, config):
        self.G = G
        self.config = config
        self.best_solution = None
        self.best_fitness = -float('inf')
    
    def random_route(self):
        """Generate a connected route using a random walk until max_route_distance is reached."""
        route = []
        total_dist = 0
        nodes = list(self.G.nodes())
        current = random.choice(nodes)
        route.append(current)
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
        return route
    
    def initialize_population(self):
        """Create a population where each individual is a list of routes (one per vehicle)."""
        population = []
        for _ in range(self.config.population_size):
            individual = []
            for _ in range(self.config.num_vehicles):
                route = self.random_route()
                individual.append(route)
            population.append(individual)
        return population
    
    def fitness(self, individual):
        """
        Fitness = total length of all routes - penalty for edge usage violation.
        For each edge used by more than three vehicles, subtract penalty_factor*(usage-3)*edge_length.
        """
        total_length = 0
        edge_usage = {}  # key: (u,v), value: count
        for route in individual:
            route_length = 0
            for u, v in zip(route[:-1], route[1:]):
                edge_data = self.G.get_edge_data(u, v)
                length = edge_data[0].get('length', 0) if edge_data else 0
                route_length += length
                edge = (u, v)
                edge_usage[edge] = edge_usage.get(edge, 0) + 1
            total_length += route_length
        
        penalty = 0
        for edge, count in edge_usage.items():
            if count > 3:
                edge_data = self.G.get_edge_data(edge[0], edge[1])
                length = edge_data[0].get('length', 0) if edge_data else 0
                penalty += self.config.penalty_factor * (count - 3) * length
        return total_length - penalty
    
    def selection(self, population):
        """Tournament selection: for each new slot, pick two random individuals and choose the one with higher fitness."""
        new_population = []
        for _ in range(len(population)):
            a, b = random.sample(population, 2)
            new_population.append(a if self.fitness(a) > self.fitness(b) else b)
        return new_population
    
    def crossover(self, parent1, parent2):
        """
        For each vehicle route, with probability crossover_rate, perform one-point crossover.
        Then repair connectivity by inserting a shortest path between mismatched nodes if needed.
        """
        child = []
        for route1, route2 in zip(parent1, parent2):
            if len(route1) < 2 or len(route2) < 2 or random.random() > self.config.crossover_rate:
                child.append(route1.copy())
            else:
                cut1 = random.randint(1, len(route1)-1)
                cut2 = random.randint(1, len(route2)-1)
                new_route = route1[:cut1] + route2[cut2:]
                repaired_route = [new_route[0]]
                for i in range(1, len(new_route)):
                    prev = repaired_route[-1]
                    curr = new_route[i]
                    if not self.G.has_edge(prev, curr):
                        try:
                            sp = nx.shortest_path(self.G, prev, curr, weight='length')
                            repaired_route.extend(sp[1:])
                        except nx.NetworkXNoPath:
                            repaired_route.append(curr)
                    else:
                        repaired_route.append(curr)
                child.append(repaired_route)
        return child
    
    def mutate(self, individual):
        """
        Mutate one vehicle route in the individual.
        For mutation, choose two indices and re-route the segment with the shortest path.
        """
        ind = [r.copy() for r in individual]
        vehicle_idx = random.randint(0, self.config.num_vehicles - 1)
        route = ind[vehicle_idx]
        if len(route) < 3:
            return ind
        i = random.randint(0, len(route)-3)
        j = random.randint(i+2, len(route)-1)
        try:
            sp = nx.shortest_path(self.G, route[i], route[j], weight='length')
            new_route = route[:i+1] + sp[1:] + route[j+1:]
            ind[vehicle_idx] = new_route
        except nx.NetworkXNoPath:
            pass
        return ind
    
    def run(self):
        population = self.initialize_population()
        best = None
        best_fit = -float('inf')
        # Wrap the generation loop with tqdm for terminal progress visualization
        for gen in tqdm(range(self.config.num_generations), desc="GA Optimization Progress"):
            # Evaluate fitness and record the best individual so far
            for ind in population:
                fit = self.fitness(ind)
                if fit > best_fit:
                    best_fit = fit
                    best = ind
            selected = self.selection(population)
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < len(selected) else selected[0]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                if random.random() < self.config.mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < self.config.mutation_rate:
                    child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population[:self.config.population_size]
        self.best_solution = best
        self.best_fitness = best_fit
        return best, best_fit

def solution_to_geojson(G, solution):
    """Convert an individual's set of routes into a GeoJSON MultiLineString for each vehicle."""
    features = []
    for vid, route in enumerate(solution):
        if len(route) < 2:
            continue
        lines = []
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                geom = edge_data[0].get('geometry',
                    LineString([
                        (G.nodes[u]['x'], G.nodes[u]['y']),
                        (G.nodes[v]['x'], G.nodes[v]['y'])
                    ])
                )
                lines.append(geom)
        if lines:
            features.append({
                "type": "Feature",
                "properties": {"vehicle": vid+1},
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [list(line.coords) for line in lines]
                }
            })
    return {"type": "FeatureCollection", "features": features}

def main():
    st.set_page_config(layout="wide")
    st.title("🚔 City-Wide Vehicle Coverage (GA Optimized)")
    
    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles", 3, 100, 50)
        population_size = st.slider("Population Size", 10, 100, 50)
        num_generations = st.slider("Generations", 10, 200, 50)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.2)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
        run_btn = st.button("Run Optimization")
    
    if run_btn:
        config = GAConfig()
        config.num_vehicles = num_vehicles
        config.population_size = population_size
        config.num_generations = num_generations
        config.mutation_rate = mutation_rate
        config.crossover_rate = crossover_rate
        
        with st.spinner("Running GA optimization..."):
            G = load_road_network("Coimbatore, India")
            optimizer = RouteOptimizerGA(G, config)
            best, best_fit = optimizer.run()
            geojson_data = solution_to_geojson(G, best)
            
            m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
            m.add_geojson(geojson_data, layer_name="Optimized Routes")
            st.success(f"Best Fitness: {best_fit:.1f}")
            m.to_streamlit()

if __name__ == "__main__":
    main()
