import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

def shortest_path_draw(graph, route_nodes):
    # Ensure the node IDs match the graph's node ID types
    graph_nodes = list(graph.nodes)

    # If the first node is a string, convert the route nodes to strings
    if isinstance(graph_nodes[0], str):
        route_nodes = [str(node) for node in route_nodes]
    else:
        route_nodes = [int(node) for node in route_nodes]

    # Calculate the shortest path from the first node to the last node
    start_node = route_nodes[0]
    end_node = route_nodes[-1]

    shortest_path = nx.shortest_path(graph, source=start_node, target=end_node, weight='length')

    # Plot the graph with OSMnx
    fig, ax = ox.plot_graph(graph, node_size=10, node_color='red', edge_color='gray', edge_linewidth=0.8, bgcolor='white', show=False, close=False)

    # Highlight the shortest path on the graph
    ox.plot_graph_route(graph, shortest_path, route_linewidth=3, node_size=50, node_color='red', route_color='blue', route_alpha=1, orig_dest_node_color='green', orig_dest_node_size=100, ax=ax)


def deviated_path_draw(graph, route_nodes):
    # Ensure the node IDs match the graph's node ID types
    graph_nodes = list(graph.nodes)

    # If the first node is a string, convert the route nodes to strings
    if isinstance(graph_nodes[0], str):
        route_nodes = [str(node) for node in route_nodes]
    else:
        route_nodes = [int(node) for node in route_nodes]

    # Plot the graph with OSMnx
    fig, ax = ox.plot_graph(graph, node_size=10, node_color='red', edge_color='gray', edge_linewidth=0.8, bgcolor='white', show=False, close=False)

    # Highlight the deviated path on the graph
    ox.plot_graph_route(graph, route_nodes, route_linewidth=3, node_size=50, node_color='red', route_color='red', route_alpha=1, orig_dest_node_color='green', orig_dest_node_size=100, ax=ax)

# Entry point of the script
if __name__ == "__main__":
    # Load the graph using OSMnx
    graph = ox.load_graphml('./map/graph.graphml')

    # Extract the node IDs for the route (as integers)
    route_nodes = [9457107118, 9457095410, 9457095395, 9457258212, 9457227827, 9443710189, 2614945489, 9457095207, 2614552637, 2613601291, 2613601289, 2613601282, 11250456423, 5297589434, 5297589433, 5297589435, 2613601271, 1471100885, 2613601368]
    # Call the shortest_path_draw function with the graph and route nodes
    shortest_path_draw(graph, route_nodes)
    # Call the deviated_path_draw function with the graph and route nodes
    deviated_path_draw(graph, route_nodes)