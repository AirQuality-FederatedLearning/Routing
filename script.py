import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import random
from shapely.geometry import LineString, MultiLineString
from datetime import datetime
from tqdm import tqdm

# Configure OSMnx caching and logging
ox.settings.use_cache = True
ox.settings.log_console = True

class RoadNetwork:
    def __init__(self, place="Coimbatore, India", cache_dir="cache"):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        safe_place = place.replace(',', '').replace(' ', '_')
        self.cache_file = os.path.join(cache_dir, f"{safe_place}_graph.graphml")
        
        if os.path.exists(self.cache_file):
            print(f"Loading cached graph from {self.cache_file}")
            self.G = ox.load_graphml(self.cache_file)
            try:
                self.G = nx.relabel_nodes(self.G, lambda x: int(x))
            except Exception as e:
                print("Node relabeling to int failed, keeping original keys:", e)
        else:
            print(f"Downloading and processing graph for {place}")
            self.G = ox.graph_from_place(
                place,
                network_type="drive",
                simplify=True,
                retain_all=False,
                truncate_by_edge=True
            )
            self.G = ox.add_edge_speeds(self.G)
            self.G = ox.add_edge_travel_times(self.G)
            print(f"Saving graph to {self.cache_file}")
            ox.save_graphml(self.G, self.cache_file)
        self.nodes = list(self.G.nodes())
        self.edges = ox.graph_to_gdfs(self.G, nodes=False).reset_index()

    def get_path_geometry(self, route_nodes):
        lines = []
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            try:
                edge_data = self.G.get_edge_data(u, v)
                if edge_data is not None:
                    data = edge_data[0]
                    if 'geometry' in data:
                        lines.append(data['geometry'])
                    else:
                        line = LineString([
                            (self.G.nodes[u]['x'], self.G.nodes[u]['y']),
                            (self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                        ])
                        lines.append(line)
            except Exception:
                continue
        if not lines:
            return None
        return MultiLineString(lines)

    def get_edge_length(self, u, v):
        try:
            edge_data = self.G.get_edge_data(u, v)
            if edge_data is not None:
                return edge_data[0].get('length', 0)
        except Exception:
            return 0
        return 0

class GAConfig:
    def __init__(self):
        self.mutation_rate = 0.1
        # Total iterations for Tabu Search (adjustable via sidebar, e.g., minimum 2, 20, 100, etc.)
        self.num_iterations = 20
        self.num_vehicles = 3
        self.max_distance = 15000  # in meters
        self.coverage_radius = 500  # in meters

class RouteOptimizer:
    def __init__(self):
        self.road_net = RoadNetwork("Coimbatore, India")
        self.config = GAConfig()
        self.history = []
        self.load_history()

    def save_history(self):
        with open("route_history.json", "w") as f:
            json.dump(self.history, f, default=str)

    def load_history(self):
        try:
            with open("route_history.json", "r") as f:
                self.history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = []

    def solution_hash(self, solution):
        """Convert the solution (list of routes) into a hashable tuple of tuples."""
        return tuple(tuple(route) for route in solution)

    def get_route_length(self, route):
        length = 0
        for u, v in zip(route[:-1], route[1:]):
            length += self.road_net.get_edge_length(u, v)
        return length

    def generate_valid_route(self):
        current_node = random.choice(self.road_net.nodes)
        route = [current_node]
        total_dist = 0
        while total_dist < self.config.max_distance:
            successors = list(self.road_net.G.successors(current_node))
            if not successors:
                break
            next_node = random.choice(successors)
            edge_length = self.road_net.get_edge_length(current_node, next_node)
            if total_dist + edge_length > self.config.max_distance:
                break
            route.append(next_node)
            total_dist += edge_length
            current_node = next_node
        return route

    def calculate_coverage(self, routes):
        buffers = []
        for route in routes:
            if len(route) < 2:
                continue
            path_geom = self.road_net.get_path_geometry(route)
            if path_geom is not None:
                gseries = gpd.GeoSeries([path_geom], crs="EPSG:4326")
                buffer_geom = gseries.buffer(self.config.coverage_radius).iloc[0]
                buffers.append(buffer_geom)
        if buffers:
            union_geom = gpd.GeoSeries(buffers, crs="EPSG:4326").to_crs(epsg=3857).unary_union
            return union_geom.area
        return 0

    def fitness(self, individual):
        return self.calculate_coverage(individual)

    def extend_route(self, route):
        current_node = route[-1]
        total_length = self.get_route_length(route)
        successors = list(self.road_net.G.successors(current_node))
        if not successors:
            return route
        next_node = random.choice(successors)
        edge_length = self.road_net.get_edge_length(current_node, next_node)
        if total_length + edge_length <= self.config.max_distance:
            route.append(next_node)
        return route

    def shorten_route(self, route):
        if len(route) > 2:
            idx = random.randint(1, len(route) - 1)
            del route[idx]
        return route

    def mutate(self, individual):
        mutated = [r.copy() for r in individual]
        vehicle_idx = random.randint(0, self.config.num_vehicles - 1)
        if len(mutated[vehicle_idx]) < 2:
            return mutated
        mutation_type = random.choice(['reroute', 'extend', 'shorten'])
        if mutation_type == 'reroute' and len(mutated[vehicle_idx]) >= 3:
            start = random.randint(0, len(mutated[vehicle_idx]) - 2)
            end = random.randint(start + 1, len(mutated[vehicle_idx]) - 1)
            try:
                new_segment = nx.shortest_path(
                    self.road_net.G,
                    mutated[vehicle_idx][start],
                    mutated[vehicle_idx][end],
                    weight='length'
                )
                mutated[vehicle_idx] = (
                    mutated[vehicle_idx][:start] +
                    new_segment +
                    mutated[vehicle_idx][end+1:]
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        elif mutation_type == 'extend':
            mutated[vehicle_idx] = self.extend_route(mutated[vehicle_idx])
        elif mutation_type == 'shorten':
            mutated[vehicle_idx] = self.shorten_route(mutated[vehicle_idx])
        return mutated

    def routes_to_geojson(self, routes):
        features = []
        for vid, route in enumerate(routes):
            if len(route) < 2:
                continue
            path_geom = self.road_net.get_path_geometry(route)
            if path_geom is None:
                continue
            if isinstance(path_geom, LineString):
                geom_type = "LineString"
                coordinates = list(path_geom.coords)
            else:
                geom_type = "MultiLineString"
                coordinates = []
                if hasattr(path_geom, "geoms"):
                    for line in path_geom.geoms:
                        coordinates.append(list(line.coords))
                else:
                    coordinates = list(path_geom.coords)
            feature = {
                "type": "Feature",
                "properties": {"vehicle": vid + 1},
                "geometry": {
                    "type": geom_type,
                    "coordinates": coordinates
                }
            }
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}

    def run_tabu_search(self):
        num_iterations = self.config.num_iterations
        tabu_size = 10
        num_neighbors = 10

        current_solution = [self.generate_valid_route() for _ in range(self.config.num_vehicles)]
        current_fitness = self.fitness(current_solution)
        best_solution = current_solution
        best_fitness = current_fitness

        # Use a list as a FIFO tabu list.
        tabu_list = [self.solution_hash(current_solution)]

        for i in tqdm(range(num_iterations), desc="Tabu Search"):
            neighbors = []
            for _ in range(num_neighbors):
                candidate = self.mutate(current_solution)
                candidate_fit = self.fitness(candidate)
                neighbors.append((candidate, candidate_fit))
            # Filter out candidates whose hash is in the tabu list.
            allowed = [(cand, fit) for cand, fit in neighbors if self.solution_hash(cand) not in tabu_list]
            if not allowed:
                continue
            candidate, candidate_fit = max(allowed, key=lambda x: x[1])
            current_solution = candidate
            current_fitness = candidate_fit
            if candidate_fit > best_fitness:
                best_solution = candidate
                best_fitness = candidate_fit
            # Update tabu list
            tabu_list.append(self.solution_hash(current_solution))
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

        result = {
            "timestamp": datetime.now().isoformat(),
            "config": vars(self.config),
            "routes": best_solution,
            "fitness": best_fitness,
            "geometry": self.routes_to_geojson(best_solution)
        }
        self.history.append(result)
        self.save_history()
        return result

def main():
    st.set_page_config(layout="wide")
    st.title("🚔 Police Route Optimization Dashboard (Tabu Search)")
    
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = RouteOptimizer()
    
    with st.sidebar:
        st.header("⚙️ Tabu Search Configuration")
        cfg = st.session_state.optimizer.config
        
        cfg.mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, cfg.mutation_rate)
        cfg.num_vehicles = st.slider("Number of Vehicles", 1, 10, cfg.num_vehicles)
        cfg.max_distance = st.slider("Max Distance (meters)", 5000, 30000, cfg.max_distance)
        cfg.coverage_radius = st.slider("Coverage Radius (meters)", 100, 1000, cfg.coverage_radius)
        cfg.num_iterations = st.slider("Total Iterations", 2, 200, cfg.num_iterations)
        
        if st.button("🌅 Run New Day Optimization"):
            st.session_state.result = st.session_state.optimizer.run_tabu_search()
            st.rerun()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🗺️ Current Day Optimization")
        if 'result' in st.session_state:
            m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
            m.add_geojson(st.session_state.result["geometry"], layer_name="Routes")
            m.to_streamlit()
        else:
            st.info("Run the optimization using the sidebar controls.")
    
    with col2:
        st.header("📅 Optimization History")
        if st.session_state.optimizer.history:
            history_df = pd.DataFrame([{
                "Date": h["timestamp"],
                "Fitness": h["fitness"],
                "Vehicles": h["config"]["num_vehicles"],
                "Distance": h["config"]["max_distance"]
            } for h in st.session_state.optimizer.history])
            history_df["Date"] = pd.to_datetime(history_df["Date"])
            history_df = history_df.sort_values("Date")
            st.line_chart(history_df.set_index("Date")["Fitness"])
            
            selected_idx = st.selectbox("View Historical Day",
                                        options=range(len(st.session_state.optimizer.history)),
                                        format_func=lambda x: f"Day {x+1}")
            
            m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)
            m.add_geojson(
                st.session_state.optimizer.history[selected_idx]["geometry"],
                layer_name="Historical Routes"
            )
            m.to_streamlit()
        else:
            st.info("No historical data available.")

if __name__ == "__main__":
    main()
