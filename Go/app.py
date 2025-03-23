import streamlit as st
import leafmap.kepler as leafmap
import json
import matplotlib.pyplot as plt
import numpy as np

def get_route_bounds(coordinates):
    """Calculate bounding box for a route"""
    lats = [coord[1] for line in coordinates for coord in line]
    lons = [coord[0] for line in coordinates for coord in line]
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons)
    }

def get_color_gradient(value, cmap_name="viridis"):
    """Get a color from a matplotlib colormap"""
    cmap = plt.get_cmap(cmap_name)
    return cmap(value)[:3]

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color code"""
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255),
    )

def main():
    st.set_page_config(layout="wide")
    st.title("🚔 Smart Police Patrol Route Optimization")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        selected_vehicle = st.selectbox("Select Vehicle", options=[i for i in range(1, 51)], index=0)
        gradient_type = st.selectbox("Color Gradient", 
                                   ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])
        show_start_points = st.checkbox("Show Police Stations", value=True)
    
    # Load and process data
    if st.button("Load Patrol Routes"):
        with open("routes.geojson") as f:
            data = json.load(f)
        
        selected_bounds = None
        start_points = []
        features = []
        
        for idx, feature in enumerate(data['features']):
            vehicle_num = idx + 1
            color = rgb_to_hex(get_color_gradient(vehicle_num / 5, gradient_type.lower()))
            
            if vehicle_num == selected_vehicle:
                bounds = get_route_bounds(feature['geometry']['coordinates'])
                selected_bounds = bounds
                start_coord = feature['geometry']['coordinates'][0][0]
                start_points.append(start_coord)
                feature['properties']['stroke'] = "#FF0000"
                feature['properties']['stroke-width'] = 5
            
            feature['properties']['stroke'] = color
            feature['properties']['stroke-width'] = 3
            features.append(feature)
        
        # Create map
        if selected_bounds:
            center_lat = (selected_bounds['min_lat'] + selected_bounds['max_lat']) / 2
            center_lon = (selected_bounds['min_lon'] + selected_bounds['max_lon']) / 2
            zoom = 13
        else:
            center_lat, center_lon, zoom = 11.0168, 76.9558, 12
        
        m = leafmap.Map(center=[center_lat, center_lon], zoom=zoom, height=800)
        
        # Add routes
        for feature in features:
            m.add_geojson(feature, layer_name=f"Vehicle {features.index(feature)+1}")
        
        # Add police stations using GeoJSON
        if show_start_points and start_points:
            station_features = []
            for idx, (lon, lat) in enumerate(start_points):
                station_features.append({
                    "type": "Feature",
                    "properties": {
                        "name": f"Police Station {idx+1}",
                        "marker-color": "#FF0000",
                        "marker-size": "medium"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    }
                })
            
            stations_geojson = {
                "type": "FeatureCollection",
                "features": station_features
            }
            
            m.add_geojson(
                stations_geojson,
                layer_name="Police Stations",
                point_style={
                    "radius": 8,
                    "fillColor": "#FF0000",
                    "color": "#000000",
                    "weight": 1,
                    "opacity": 1,
                    "fillOpacity": 0.8
                }
            )
        
        # Display map
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("## Patrol Route Visualization")
            m.to_streamlit()
        
        with col2:
            st.write("### Legend")
            st.markdown("""
                <style>
                    .legend { padding: 10px; background: white; border-radius: 5px; }
                    .legend-item { display: flex; align-items: center; margin: 5px 0; }
                    .legend-color { width: 20px; height: 20px; margin-right: 10px; }
                </style>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background:#FF0000"></div>
                        <span>Selected Vehicle</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:#888888"></div>
                        <span>Other Vehicles</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:#FF0000"></div>
                        <span>Police Station</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()