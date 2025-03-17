import streamlit as st
import leafmap.foliumap as leafmap
import json

def load_geojson(filename=r'C:\Users\DELL\Documents\Amrita\4th year\ArcGis\solution_cache\aco_solution.geojson'):
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("GeoJSON file not found. Run the Go optimizer first.")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("🚔 Vehicle Route Optimization Viewer")

    data = load_geojson()

    if data:
        vehicle_features = {}
        for feature in data['features']:
            vid = feature['properties']['vehicle']
            if vid not in vehicle_features:
                vehicle_features[vid] = {
                    "type": "FeatureCollection",
                    "features": []
                }
            vehicle_features[vid]["features"].append(feature)

        m = leafmap.Map(center=[11.0168, 76.9558], zoom=12)

        with st.sidebar:
            st.header("🚗 Select Vehicles")
            selected_vehicles = []
            for vid in sorted(vehicle_features.keys()):
                if st.checkbox(f"Vehicle {vid}", value=True):
                    selected_vehicles.append(vid)

        for vid in selected_vehicles:
            color = vehicle_features[vid]['features'][0]['properties']['color']
            m.add_geojson(
                vehicle_features[vid],
                layer_name=f"Vehicle {vid}",
                style={
                    'color': color,
                    'weight': 4,
                    'opacity': 0.8
                }
            )

        m.add_layer_control()

        st.write("### Optimized Vehicle Routes:")
        m.to_streamlit(height=700)

if __name__ == "__main__":
    main()
