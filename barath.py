import streamlit as st
import math
import random
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta
import numpy as np

# -----------------------------
# 1. Define Government Baselines and Routes
# -----------------------------
baselines = {
    "pm2.5": 35.0,   # micrograms/m³
    "pm10": 50.0,    # micrograms/m³
    "o3": 100.0,     # parts per billion (ppb)
    "no2": 40.0,     # ppb
    "so2": 20.0,     # ppb
    "co": 4.0,       # parts per million (ppm)
}

# Predefined routes for devices
DEVICE_ROUTES = {
    "device_1": {
        "name": "Sitra to Peelamedu",
        "waypoints": [
            (10.9915, 76.9516),  # Sitra
            (11.0054, 76.9618),  # R.S. Puram
            (11.0168, 76.9558),  # Gandhipuram
            (11.0245, 76.9601)   # Peelamedu
        ],
        "start_time": datetime(2024, 3, 1, 8, 0)  # 8 AM
    },
    "device_2": {
        "name": "Ukkadam to Thudiyalur",
        "waypoints": [
            (11.0045, 76.9631),  # Ukkadam
            (11.0123, 76.9555),  # Town Hall
            (11.0234, 76.9488),  # Thudiyalur
        ],
        "start_time": datetime(2024, 3, 1, 9, 0)  # 9 AM
    }
}

# -----------------------------
# 2. Core Functions
# -----------------------------
def generate_path_data(device_id, num_points=20):
    """Generate path data with timestamps and measurements between waypoints."""
    route = DEVICE_ROUTES[device_id]
    waypoints = route["waypoints"]
    current_time = route["start_time"]
    
    measurements = []
    time_step = timedelta(minutes=15)
    
    # Generate interpolated points between waypoints
    for i in range(len(waypoints) - 1):
        start_lat, start_lon = waypoints[i]
        end_lat, end_lon = waypoints[i + 1]
        
        # Generate intermediate points
        for frac in np.linspace(0, 1, num_points // (len(waypoints) - 1)):
            lat = start_lat + frac * (end_lat - start_lat)
            lon = start_lon + frac * (end_lon - start_lon)
            
            # Generate sensor data with temporal variation
            hour_factor = current_time.hour / 24
            measurements.append({
                "device_id": device_id,
                "timestamp": current_time,
                "gps": (lat, lon),
                "params": {
                    "pm2.5": random.uniform(20 + 20 * hour_factor, 60 * hour_factor),
                    "pm10": random.uniform(30 + 30 * hour_factor, 70 * hour_factor),
                    "o3": random.uniform(80 - 40 * hour_factor, 120 * hour_factor),
                    "no2": random.uniform(30 + 20 * hour_factor, 50 * hour_factor),
                    "so2": random.uniform(10 + 10 * hour_factor, 30 * hour_factor),
                    "co": random.uniform(2 + 2 * hour_factor, 6 * hour_factor),
                }
            })
            current_time += time_step
            
    return measurements

def check_exceedances(measurement, baselines):
    """Check sensor readings against baseline values."""
    return {
        pollutant: value
        for pollutant, value in measurement["params"].items()
        if value > baselines.get(pollutant, float('inf'))
    }

def haversine(coord1, coord2):
    """Compute the great-circle distance (km) between two GPS coordinates."""
    R = 6371.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2)**2 + 
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# -----------------------------
# 3. Main Application
# -----------------------------
st.title("Mobile AQI Monitoring Station Analysis")

# Generate device paths
all_measurements = []
for device_id in DEVICE_ROUTES:
    all_measurements.extend(generate_path_data(device_id))

# Convert to DataFrame
df = pd.DataFrame(all_measurements)
df['exceedances'] = df.apply(lambda x: check_exceedances(x, baselines), axis=1)
df['exceed_count'] = df['exceedances'].apply(len)
df[['lat', 'lon']] = pd.DataFrame(df['gps'].tolist(), index=df.index)

# Time range selector
min_time = df['timestamp'].min()
max_time = df['timestamp'].max()
# Use the slider to select a datetime range
selected_time = st.slider(
    "Select analysis time window",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time)
)

time_filtered = df[
    (df['timestamp'] >= selected_time[0]) &
    (df['timestamp'] <= selected_time[1])
]

# Clustering
cluster_threshold_km = 1.5
clusters = []
visited = set()

for idx, row in time_filtered.iterrows():
    if idx in visited:
        continue
    
    cluster = [row]
    visited.add(idx)
    
    for _, other_row in time_filtered.iterrows():
        if other_row.name in visited:
            continue
        
        distance = haversine(row['gps'], other_row['gps'])
        if distance < cluster_threshold_km:
            cluster.append(other_row)
            visited.add(other_row.name)
    
    clusters.append(cluster)

# -----------------------------
# 4. Visualization
# -----------------------------
st.subheader("Mobile Device Paths and Pollution Clusters")

# Path layer
path_layer = pdk.Layer(
    "PathLayer",
    data=[{
        "path": group[['lat', 'lon']].values.tolist(),
        "name": device_id,
        "color": [255, 0, 0] if device_id == "device_1" else [0, 0, 255]
    } for device_id, group in df.groupby('device_id')],
    get_path="path",
    get_color="color",
    get_width=5,
    pickable=True
)

# Cluster layer
cluster_data = []
for i, cluster in enumerate(clusters):
    for point in cluster:
        cluster_data.append({
            "lat": point['lat'],
            "lon": point['lon'],
            "cluster": i + 1,
            "exceed_count": point['exceed_count']
        })

cluster_layer = pdk.Layer(
    "ScatterplotLayer",
    data=cluster_data,
    get_position=["lon", "lat"],
    get_radius=100,
    get_fill_color=[255, 140, 0, 200],
    pickable=True
)

# Heatmap layer
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=cluster_data,
    get_position=["lon", "lat"],
    get_weight="exceed_count",
    radius_pixels=50,
    intensity=1.0,
    threshold=0.8
)

# Deck setup
view_state = pdk.ViewState(
    latitude=11.0168,
    longitude=76.9558,
    zoom=12,
    pitch=50
)

tooltip = {
    "html": "<b>Cluster {cluster}</b><br/>Exceedances: {exceed_count}",
    "style": {"color": "white"}
}

st.pydeck_chart(pdk.Deck(
    layers=[path_layer, cluster_layer, heatmap_layer],
    initial_view_state=view_state,
    tooltip=tooltip
))

# -----------------------------
# 5. Analysis Results
# -----------------------------
st.subheader("Cluster Analysis")
for i, cluster in enumerate(clusters):
    total_exceed = sum(p['exceed_count'] for p in cluster)
    avg_exceed = total_exceed / len(cluster)
    area = len(cluster) * math.pi * (cluster_threshold_km**2)
    
    st.write(f"""
    **Cluster {i+1}**
    - Coverage area: {area:.2f} km²
    - Total exceedances: {total_exceed}
    - Average exceedances per reading: {avg_exceed:.2f}
    """)

st.subheader("Device Path Details")
for device_id, group in df.groupby('device_id'):
    st.write(f"""
    **{device_id}** ({DEVICE_ROUTES[device_id]['name']})
    - Start: {group['timestamp'].min().strftime('%H:%M')}
    - End: {group['timestamp'].max().strftime('%H:%M')}
    - Total readings: {len(group)}
    - Average exceedances: {group['exceed_count'].mean():.2f}
    """)