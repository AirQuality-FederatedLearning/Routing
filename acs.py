import os
import streamlit as st
import leafmap.kepler as leafmap
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
import random
import numpy as np
import time
from tqdm import trange

ox.settings.use_cache = True
ox.settings.log_console = True

###############################################################################
# 1) LOAD / CACHING THE ROAD NETWORK
###############################################################################
def load_road_network(place="Coimbatore, India", cache_dir="road_cache"):
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
    try:
        gdf_police = ox.features_from_place(place, tags={"amenity":"police"})
    except Exception as e:
        st.warning(f"Could not fetch police stations: {e}")
        return []
    if gdf_police.empty:
        st.info("No police stations found. Falling back to random start nodes.")
        return []
    station_nodes=[]
    for idx, row in gdf_police.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        pt = geom.centroid if geom.geom_type != 'Point' else geom
        node_id = ox.distance.nearest_nodes(G, pt.x, pt.y)
        station_nodes.append(node_id)
    return list(set(station_nodes))

###############################################################################
# 3) ANT COLONY SYSTEM (ACS) CONFIG AND OPTIMIZER
###############################################################################
class ACSConfig:
    def __init__(self):
        self.num_vehicles = 5
        self.num_iterations= 50
        self.alpha = 1.0
        self.beta = 2.0
        # Evap
        self.evaporation_rate = 0.1
        self.Q = 100

        self.max_route_distance = 15000
        self.penalty_factor = 10000
        self.coverage_penalty_factor = 8000

        # ACS-specific
        self.phi_local = 0.1   # local update factor
        self.tau0 = 1.0        # base pheromone
        self.q0 = 0.9          # pseudo-random threshold for exploitation

class ACSOptimizer:
    def __init__(self, G, config, station_nodes=None):
        self.G = G
        self.config = config
        self.station_nodes = station_nodes if station_nodes else []

        self.edge_length_cache={}
        self.adjacency_cache={}
        self._build_edge_length_cache()
        self._build_adjacency_cache()

        self.all_edges = set(self.edge_length_cache.keys())
        # Pheromone init
        self.pheromone={ (u,v): config.tau0 for (u,v) in self.edge_length_cache.keys() }

        self.best_solution=None
        self.best_fitness=-float('inf')

    def _build_edge_length_cache(self):
        for u,v,data in self.G.edges(data=True):
            length = data.get('length',0)
            self.edge_length_cache[(u,v)] = length

    def _build_adjacency_cache(self):
        for n in self.G.nodes():
            self.adjacency_cache[n]= list(self.G.successors(n))

    def choose_next_node(self, current):
        neighbors = self.adjacency_cache[current]
        if not neighbors:
            return None
        # pseudo-random proportional rule
        q = random.random()
        if q < self.config.q0:
            # Exploitation: pick best next
            best_node=None
            best_value=-1
            for nbr in neighbors:
                length = self.edge_length_cache.get((current,nbr),1)
                tau = self.pheromone[(current,nbr)]**self.config.alpha
                eta = (1.0/length)**self.config.beta
                val = tau*eta
                if val>best_value:
                    best_value= val
                    best_node = nbr
            return best_node
        else:
            # Exploration: choose via standard roulette
            desirabilities=[]
            total=0.0
            for nbr in neighbors:
                length=self.edge_length_cache.get((current,nbr),1)
                tau = self.pheromone[(current,nbr)]**self.config.alpha
                eta = (1.0/length)**self.config.beta
                d= tau*eta
                desirabilities.append((nbr,d))
                total+= d
            if total==0:
                return random.choice(neighbors)
            r = random.random()*total
            s=0.0
            for (nbr,val) in desirabilities:
                s+= val
                if s>=r:
                    return nbr
            return desirabilities[-1][0]

    def local_update(self, u,v):
        # local pheromone update
        tau_uv = self.pheromone[(u,v)]
        phi = self.config.phi_local
        tau0= self.config.tau0
        new_tau = (1-phi)*tau_uv + phi*tau0
        self.pheromone[(u,v)] = new_tau

    def build_route(self):
        start_node = random.choice(self.station_nodes) if self.station_nodes else random.choice(list(self.G.nodes()))
        route=[start_node]
        total_dist=0
        current=start_node
        while True:
            nxt = self.choose_next_node(current)
            if nxt is None:
                break
            dist = self.edge_length_cache.get((current,nxt),0)
            if total_dist+dist>self.config.max_route_distance:
                break
            route.append(nxt)
            total_dist+=dist
            # local update
            self.local_update(current, nxt)
            current=nxt
        return route

    def build_solution(self):
        return [self.build_route() for _ in range(self.config.num_vehicles)]

    def fitness(self, solution):
        total_length=0
        edge_usage={}
        covered_edges=set()
        for route in solution:
            route_len=0
            for u,v in zip(route[:-1], route[1:]):
                length=self.edge_length_cache.get((u,v),0)
                route_len+= length
                covered_edges.add((u,v))
                edge_usage[(u,v)] = edge_usage.get((u,v),0)+1
            total_length+= route_len
        overlap_penalty=0
        for (u,v),count in edge_usage.items():
            if count>3:
                length= self.edge_length_cache.get((u,v),0)
                overlap_penalty+= self.config.penalty_factor*(count-3)*length
        uncovered= self.all_edges - covered_edges
        coverage_penalty=0
        for (u,v) in uncovered:
            length=self.edge_length_cache.get((u,v),0)
            coverage_penalty+= self.config.coverage_penalty_factor*length
        return total_length - overlap_penalty - coverage_penalty

    def global_update(self, best_sol, best_fit):
        deposit = max(best_fit,1)
        for route in best_sol:
            for u,v in zip(route[:-1], route[1:]):
                old_val = self.pheromone[(u,v)]
                self.pheromone[(u,v)] = (1-self.config.evaporation_rate)*old_val + \
                                        self.config.evaporation_rate*(deposit)

    def run(self):
        for _ in trange(self.config.num_iterations, desc="ACS Iteration"):
            sol= self.build_solution()
            fit= self.fitness(sol)
            if fit> self.best_fitness:
                self.best_fitness= fit
                self.best_solution= sol
            # after each iteration we do global update using best solution
            self.global_update(self.best_solution, self.best_fitness)
        return self.best_solution, self.best_fitness

###############################################################################
# 4) CONVERT SOLUTION TO GEOJSON
###############################################################################
def solution_to_geojson(G, solution):
    colors=[
        "#e6194B","#3cb44b","#ffe119","#4363d8","#f58231",
        "#911eb4","#46f0f0","#f032e6","#bcf60c","#fabebe"
    ]
    features=[]
    for vid,route in enumerate(solution):
        if len(route)<2:
            continue
        lines=[]
        for u,v in zip(route[:-1], route[1:]):
            edge_data= G.get_edge_data(u,v)
            if edge_data:
                geom = edge_data[0].get('geometry', LineString([
                    (G.nodes[u]['x'],G.nodes[u]['y']),
                    (G.nodes[v]['x'],G.nodes[v]['y'])
                ]))
                lines.append(geom)
        if lines:
            color = colors[vid%len(colors)]
            features.append({
                "type":"Feature",
                "properties":{
                    "vehicle": vid+1,
                    "stroke": color,
                    "stroke-width":3,
                    "stroke-opacity":0.8
                },
                "geometry":{
                    "type":"MultiLineString",
                    "coordinates":[list(line.coords) for line in lines]
                }
            })
    return {"type":"FeatureCollection","features":features}

def route_to_geojson(G, route, vehicle_id):
    lines=[]
    for u,v in zip(route[:-1], route[1:]):
        edge_data= G.get_edge_data(u,v)
        if edge_data:
            geom = edge_data[0].get('geometry', LineString([
                (G.nodes[u]['x'],G.nodes[u]['y']),
                (G.nodes[v]['x'],G.nodes[v]['y'])
            ]))
            lines.append(geom)
    if lines:
        feature={
            "type":"Feature",
            "properties":{
                "vehicle":vehicle_id,
                "stroke":"#FF0000",
                "stroke-width":5,
                "stroke-opacity":1.0
            },
            "geometry":{
                "type":"MultiLineString",
                "coordinates":[list(line.coords) for line in lines]
            }
        }
        return {"type":"FeatureCollection","features":[feature]}
    else:
        return None

###############################################################################
# 5) STREAMLIT MAIN APP - ACS
###############################################################################
def main():
    st.set_page_config(layout="wide")
    st.title("🚔 City-Wide Vehicle Coverage (ACS Variant)")

    with st.sidebar:
        num_vehicles = st.slider("Number of Vehicles (Ants)",3,50,5)
        num_iterations = st.slider("Number of Iterations", 10,500,50)
        alpha= st.slider("Alpha (pheromone influence)", 0.0,5.0,1.0)
        beta= st.slider("Beta (distance influence)", 0.0,5.0,2.0)
        evap_rate= st.slider("Global Evap Rate (ρ)", 0.0,1.0,0.1)
        phi_local= st.slider("Local Update (phi)", 0.0,1.0,0.1)
        tau0= st.slider("Initial Pheromone (tau0)", 0.01,5.0,1.0)
        q0= st.slider("Pseudo-Random q0", 0.0,1.0,0.9)
        Q= st.slider("Pheromone Deposit (Q)",1,1000,100)
        coverage_penalty= st.slider("Coverage Penalty Factor",1000,20000,8000, step=1000)
        run_btn= st.button("Run ACS")

        if "acs_solution" in st.session_state:
            route_opts= ["None"]+[f"Route {i+1}" for i in range(len(st.session_state['acs_solution']))]
            selected_route= st.selectbox("Select a route to highlight", options=route_opts)
        else:
            selected_route="None"

    if run_btn:
        start_time=time.time()
        config= ACSConfig()
        config.num_vehicles=num_vehicles
        config.num_iterations=num_iterations
        config.alpha= alpha
        config.beta= beta
        config.evaporation_rate=evap_rate
        config.phi_local=phi_local
        config.tau0= tau0
        config.q0= q0
        config.Q= Q
        config.coverage_penalty_factor= coverage_penalty

        with st.spinner("Loading data and running ACS..."):
            G = load_road_network("Coimbatore, India")
            station_nodes= get_police_station_nodes(G, "Coimbatore, India")

            optimizer = ACSOptimizer(G, config, station_nodes)
            best_solution, best_fit= optimizer.run()
            geojson_data = solution_to_geojson(G, best_solution)

            st.session_state['acs_graph']= G
            st.session_state['acs_solution']= best_solution
            st.session_state['acs_fit']= best_fit
            st.session_state['acs_geojson']= geojson_data

        end_time=time.time()
        st.write(f"Time taken: {end_time - start_time:.2f} seconds.")

    if "acs_solution" in st.session_state:
        G = st.session_state['acs_graph']
        best_solution= st.session_state['acs_solution']
        best_fit = st.session_state['acs_fit']
        geojson_data= st.session_state['acs_geojson']

        boundary_gdf= ox.geocode_to_gdf("Coimbatore, India")
        m= leafmap.Map(center=[11.0168,76.9558], zoom=12)
        m.add_gdf(boundary_gdf, layer_name="Coimbatore Boundary",
                  style={"color":"#FF0000","fillOpacity":0.1})
        m.add_geojson(geojson_data, layer_name="Optimized Routes (ACS)")

        if selected_route!="None":
            try:
                idx= int(selected_route.split(" ")[1]) - 1
                if 0<=idx < len(best_solution):
                    highlight_geojson= route_to_geojson(G, best_solution[idx], idx+1)
                    if highlight_geojson:
                        m.add_geojson(highlight_geojson, layer_name=f"Vehicle {idx+1} Highlight")
                        coords=[]
                        for node in best_solution[idx]:
                            lat= G.nodes[node]['y']
                            lon= G.nodes[node]['x']
                            coords.append([lat, lon])
                        if coords:
                            lats= [pt[0] for pt in coords]
                            lons= [pt[1] for pt in coords]
                            bounds=[[min(lats), min(lons)], [max(lats), max(lons)]]
                            m.fit_bounds(bounds)
            except Exception as e:
                st.error(f"Error highlighting route: {e}")

        st.success(f"Best Fitness: {best_fit:.1f}")
        m.to_streamlit()

if __name__=="__main__":
    main()
