import os
import xml.etree.ElementTree as ET
import csv

def convert_graphml_to_csv(graphml_file, csv_file):
    # Parse the GraphML file
    tree = ET.parse(graphml_file)
    root = tree.getroot()
    
    # Namespace handling
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    
    # Create a list to hold the edges data
    edges = []
    
    # Iterate through each edge in the GraphML
    for edge in root.findall('.//graphml:edge', ns):
        source = edge.get('source')
        target = edge.get('target')
        
        # Find the length (weight) of the edge
        length_data = edge.find('graphml:data[@key="d12"]', ns)
        if length_data is not None:
            weight = length_data.text
            edges.append((source, target, weight))
    
    # Write the edges to a CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['source', 'destination', 'weight'])
        # Write data rows
        writer.writerows(edges)

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set path for the map file
    map_dir = os.path.join(script_dir, 'map')
    # Load the OSM graph data from the GraphML file
    graph_file = os.path.join(map_dir, 'graph.graphml')
    # set path for 
    csv_file = os.path.join(map_dir, 'graph.csv')

    # Call the conversion function
    convert_graphml_to_csv(graph_file, csv_file)
    print(f"Conversion complete! Data written to {csv_file}")