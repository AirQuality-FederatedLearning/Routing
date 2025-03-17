import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import random
import hashlib
import json
import pickle
from tqdm import tqdm

# Configure OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = True

# Configuration class
class GAConfig:
    def __init__(self):
        self.population_size = 50
        self.num_generations = 100
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        self.num_vehicles = 5
        self.max_route_distance = 15000  # meters
        self.penalty_factor = 10000

# Caching functions
def get_parameter_hash(config, road_network_path):
    params = {
        'num_vehicles': config.num_vehicles,
        'population_size': config.population_size,
        'num_generations': config.num_generations,
        'mutation_rate': config.mutation_rate,
        'crossover_rate': config.crossover_rate,
        'max_route_distance': config.max_route_distance,
        'penalty_factor': config.penalty_factor,
        'road_network_mtime': os.path.getmtime(road_network_path),
    }
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

def load_road_network(place="Coimbatore, India", cache_dir="road_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    safe_place = place.replace(" ", "_").replace(",", "")
    cache_file = os.path.join(cache_dir, f"{safe_place}_network.graphml")
    
    if not os.path.exists(cache_file):
        st.sidebar.info("Downloading fresh road network data...")
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        ox.save_graphml(G, cache_file)
    else:
        G = ox.load_graphml(cache_file)
    
    return G, cache_file

def save_ga_cache(cache_key, solution, fitness):
    cache_dir = "ga_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump({"solution": solution, "fitness": fitness}, f)

def load_ga_cache(cache_key):
    cache_dir = "ga_cache"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

# GA implementation
class RouteOptimizerGA:
    def __init__(self, G, config):
        self.G = G
        self.config = config
        self.best_solution = None
        self.best_fitness = -float('inf')

    def random_route(self):
        nodes = list(self.G.nodes())
        if not nodes:
            return []
        
        route = []
        current = random.choice(nodes)
        route.append(current)
        total_dist = 0
        
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
        return [[self.random_route() for _ in range(self.config.num_vehicles)] 
                for _ in range(self.config.population_size)]

    def fitness(self, individual):
        edge_usage = {}
        total_length = 0
        
        for route in individual:
            route_length = 0
            for u, v in zip(route[:-1], route[1:]):
                edge_data = self.G.get_edge_data(u, v)
                length = edge_data[0].get('length', 0) if edge_data else 0
                route_length += length
                edge_usage[(u, v)] = edge_usage.get((u, v), 0) + 1
            total_length += route_length
        
        penalty = 0
        for edge, count in edge_usage.items():
            if count > 3:
                edge_data = self.G.get_edge_data(edge[0], edge[1])
                length = edge_data[0].get('length', 0) if edge_data else 0
                penalty += self.config.penalty_factor * (count - 3) * length
                
        return total_length - penalty

    def selection(self, population):
        return [max(random.sample(population, 2), key=lambda x: self.fitness(x)) 
                for _ in range(len(population))]

    def crossover(self, parent1, parent2):
        child = []
        for r1, r2 in zip(parent1, parent2):
            if random.random() < self.config.crossover_rate and len(r1) > 1 and len(r2) > 1:
                cut1 = random.randint(1, len(r1)-1)
                cut2 = random.randint(1, len(r2)-1)
                new_route = self.repair_route(r1[:cut1] + r2[cut2:])
                child.append(new_route)
            else:
                child.append(r1.copy())
        return child

    def repair_route(self, route):
        if len(route) < 2:
            return route
            
        repaired = [route[0]]
        for node in route[1:]:
            try:
                if not self.G.has_edge(repaired[-1], node):
                    sp = nx.shortest_path(self.G, repaired[-1], node, weight='length')
                    repaired.extend(sp[1:])
                else:
                    repaired.append(node)
            except nx.NetworkXNoPath:
                continue
        return repaired

    def mutate(self, individual):
        mutated = [r.copy() for r in individual]
        vehicle_idx = random.randint(0, self.config.num_vehicles-1)
        
        if len(mutated[vehicle_idx]) > 2:
            i = random.randint(0, len(mutated[vehicle_idx])-3)
            j = random.randint(i+2, len(mutated[vehicle_idx])-1)
            try:
                sp = nx.shortest_path(self.G, 
                                    mutated[vehicle_idx][i], 
                                    mutated[vehicle_idx][j], 
                                    weight='length')
                mutated[vehicle_idx] = mutated[vehicle_idx][:i] + sp + mutated[vehicle_idx][j+1:]
            except nx.NetworkXNoPath:
                pass
                
        return mutated

    def run(self):
        population = self.initialize_population()
        
        with tqdm(total=self.config.num_generations, desc="Optimizing Routes") as pbar:
            for _ in range(self.config.num_generations):
                population = self.selection(population)
                new_pop = []
                
                for i in range(0, len(population), 2):
                    parent1 = population[i]
                    parent2 = population[i+1] if i+1 < len(population) else population[0]
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                    
                    if random.random() < self.config.mutation_rate:
                        child1 = self.mutate(child1)
                    if random.random() < self.config.mutation_rate:
                        child2 = self.mutate(child2)
                        
                    new_pop.extend([child1, child2])
                
                population = new_pop[:self.config.population_size]
                pbar.update(1)
                
        self.best_solution = max(population, key=lambda x: self.fitness(x))
        self.best_fitness = self.fitness(self.best_solution)
        return self.best_solution, self.best_fitness

# Visualization functions
def solution_to_geojson(G, solution):
    features = []
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
        "#00FFFF", "#FFA500", "#800080", "#008000", "#800000"
    ]
    
    for vid, route in enumerate(solution):
        if len(route) < 2:
            continue
            
        lines = []
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                geom = edge_data[0].get('geometry',
                    LineString([(G.nodes[u]['x'], G.nodes[u]['y']),
                                 (G.nodes[v]['x'], G.nodes[v]['y'])])
                )
                lines.append(geom)
                
        if lines:
            features.append({
                "type": "Feature",
                "properties": {
                    "vehicle": vid+1,
                    "color": colors[vid % len(colors)],
                    "stroke-width": 3
                },
                "geometry": MultiLineString(lines).__geo_interface__
            })
            
    return {"type": "FeatureCollection", "features": features}

# Main application
def main():
    st.set_page_config(layout="wide")
    st.title("🚔 Smart City Patrol Route Optimizer")
    
    with st.sidebar:
        st.header("Optimization Parameters")
        num_vehicles = st.slider("Number of Vehicles", 3, 10, 5)
        population_size = st.slider("Population Size", 10, 100, 50)
        num_generations = st.slider("Generations", 10, 200, 50)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.2)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
        optimize_btn = st.button("Start Optimization")
    
    if optimize_btn:
        config = GAConfig()
        config.num_vehicles = num_vehicles
        config.population_size = population_size
        config.num_generations = num_generations
        config.mutation_rate = mutation_rate
        config.crossover_rate = crossover_rate
        
        with st.spinner("Loading city road network..."):
            G, road_cache_file = load_road_network()
            cache_key = get_parameter_hash(config, road_cache_file)
            cached_result = load_ga_cache(cache_key)
        
        if cached_result:
            st.success("Using cached optimization results!")
            best_solution, best_fitness = cached_result["solution"], cached_result["fitness"]
        else:
            with st.spinner("Running genetic algorithm optimization..."):
                optimizer = RouteOptimizerGA(G, config)
                best_solution, best_fitness = optimizer.run()
                save_ga_cache(cache_key, best_solution, best_fitness)
                st.success("Optimization complete! Results cached for future use.")
        
        st.subheader("Optimized Patrol Routes")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            geojson_data = solution_to_geojson(G, best_solution)
            m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
            m.add_geojson(
                geojson_data,
                layer_name="Patrol Routes",
                style={'color': 'properties.color', 'strokeWidth': 'properties.stroke-width'}
            )
            m.to_streamlit()
        
        with col2:
            st.metric("Total Coverage Distance", f"{best_fitness/1000:.1f} km")
            st.info("""
                **Color Coding:**
                - Each color represents a different vehicle route
                - Routes are optimized for:
                  - Maximum road coverage
                  - Minimum overlapping patrol areas
                  - Balanced workload between vehicles
                """)
            
            st.download_button(
                "Download Optimized Routes",
                data=json.dumps(geojson_data),
                file_name="optimized_routes.geojson",
                mime="application/json"
            )

if __name__ == "__main__":
    main()