import osmnx as ox

# Step 1: Download OSM Data for IIT Kharagpur and save the graph
graph = ox.graph_from_place("IIT Kharagpur", network_type='drive')
ox.save_graphml(graph, "./map/graph.graphml")

# Step 2: Retrieve POIs and save them as GeoJSON
tags = {
    "building": True,            # All buildings
    "amenity": True,             # All amenities (schools, hospitals, etc.)
    "shop": True,                # All types of shops
    "office": True,              # Office buildings
    "tourism": True,             # Tourist attractions, landmarks, etc.
    "leisure": True,             # Leisure places like parks, sports facilities
    "historic": True,            # Historic sites
    "public_transport": True      # Bus stops, stations, etc.
}

# Retrieve all POIs based on the defined tags
pois = ox.geometries_from_place("IIT Kharagpur", tags=tags)

# Save the POIs to a GeoJSON file
pois.to_file("./map/pois.geojson", driver='GeoJSON')

print("Graph and POIs have been saved successfully!")
