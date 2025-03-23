import streamlit as st
import leafmap.kepler as leafmap
import json

def main():
    st.title("ACO + RRT* Routes")
    if st.button("Load GeoJSON"):
        with open("routes.geojson") as f:
            data = json.load(f)
        m = leafmap.Map(center=[11.0, 76.9], zoom=11)
        m.add_geojson(data, layer_name="Patrol Routes")
        st.write("## ACO + RRT* Routes")
        m.to_streamlit()

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
