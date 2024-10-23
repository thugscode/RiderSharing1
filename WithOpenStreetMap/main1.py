import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import random
import json
import os
import sys


class Map:
    def __init__(self, graph_file):
        self.graph = self.load_osm_data(graph_file)

    @staticmethod
    def load_osm_data(graph_file):
        """Load the street network from a local GraphML file."""
        try:
            G = ox.load_graphml(graph_file)
        
            # Plot the graph
            fig, ax = ox.plot_graph(G, node_size=10, node_color='red', edge_color='gray', edge_linewidth=0.8, bgcolor='white')
            
            # Get node positions (lat, lon) and add coordinate annotations to the plot
            pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
            
            for node, (x, y) in pos.items():
                coord_str = f"({y:.7f}, {x:.7f})"  # (lat, lon)
                ax.text(x, y, coord_str, fontsize=2, color='blue', ha='center', va='center')
            
            # Save the plot if save_path is provided
            save_path="./Output1/map1.png"
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Graph saved successfully at: {save_path}")
            # Display the plot
            plt.show()
            
            return G
        except Exception as e:
            print(f"Error loading OSM data: {e}")
            raise
        

    def draw_route(self, source, destination):
        """Calculate and draw the shortest path on the graph."""
        print(f"Finding route from {source} to {destination}...")
        source_node = ox.distance.nearest_nodes(self.graph, source[1], source[0])
        destination_node = ox.distance.nearest_nodes(self.graph, destination[1], destination[0])

        if source_node is None or destination_node is None:
            raise ValueError(f"Invalid nodes: {source_node}, {destination_node}")

        try:
            route = nx.shortest_path(self.graph, source_node, destination_node, weight='length')
            return route
        except nx.NetworkXNoPath:
            print(f"No path between {source} and {destination}.")
            return []
        except Exception as e:
            print(f"Error calculating route: {e}")
            raise


class Driver:
    def __init__(self, map_obj, id, source, destination, seats, threshold):
        self.map_obj = map_obj
        self.id = id  # Added driver ID
        self.source = source
        self.destination = destination
        self.seats = seats
        self.threshold = threshold
        self.route = None


class Rider:
    def __init__(self, map_obj, id, source, destination):
        self.map_obj = map_obj
        self.id = id  # Added rider ID
        self.source = source
        self.destination = destination
        self.matched_driver = None



class EligibilityRiderMatrix:
    def __init__(self, map_obj):
        self.map_obj = map_obj
        self.ER = None 
        self.offers = None

    def calculate(self, drivers, riders):
        num_drivers = len(drivers)
        num_riders = len(riders)

        self.ER = np.zeros((num_drivers, num_riders), dtype=int)
        self.offers = np.zeros(num_riders, dtype=int)

        for i, driver in enumerate(drivers):
            SP, sp_length = self.shortest_path_distance(driver.source, driver.destination)
            t = driver.threshold
            MP = sp_length * (1 + (t / 100))
            # print(f'{driver.id}')
            for j, rider in enumerate(riders):
                DP = self.calculate_deviated_path(driver, rider)
                if DP <= MP:
                    self.ER[i][j] = 1
                    # print(f'{rider.id} DP={DP} MP={MP} SP={sp_length}')           
        self.update_offers()
        
             
    def shortest_path_distance(self, source, target):
        """Calculate the shortest path distance between two points."""
        try:
            source_node = ox.distance.nearest_nodes(self.map_obj.graph, source[1], source[0])
            target_node = ox.distance.nearest_nodes(self.map_obj.graph, target[1], target[0])

            # Calculate shortest path length using 'length' attribute
            path_length = nx.shortest_path_length(self.map_obj.graph, source=source_node, target=target_node, weight='length')
            return nx.shortest_path(self.map_obj.graph, source=source_node, target=target_node, weight='length'), path_length
        except nx.NetworkXNoPath:
            return None, float('inf')
        except Exception as e:
            print(f"Error in shortest path calculation: {e}")
            return None, float('inf')

    def calculate_deviated_path(self, driver, rider):
        """Calculate the deviated path distance by considering the rider's route."""
        DP1, dp1_length = self.shortest_path_distance(driver.source, rider.source)
        DP2, dp2_length = self.shortest_path_distance(rider.source, rider.destination)
        DP3, dp3_length = self.shortest_path_distance(rider.destination, driver.destination)

        # Total deviated path length
        DP = dp1_length + dp2_length + dp3_length
        return DP

    def update_offers(self):
        """Update the offers based on the eligibility matrix."""
        self.offers = np.sum(self.ER, axis=0)

    def assign_riders_to_drivers(self, drivers, riders):
        DP_assigned = {driver.id: {'driver_path': [], 'riders': []} for driver in drivers}
        while np.sum(self.offers) > 0:
            non_zero_offers = self.offers[self.offers > 0]
            if non_zero_offers.size == 0:
                break

            Min_offer = np.min(non_zero_offers)
            Min_offer_set = np.where(self.offers == Min_offer)[0]

            r_selected = Min_offer_set[0] if len(Min_offer_set) == 1 else random.choice(Min_offer_set)
            eligible_drivers = np.where(self.ER[:, r_selected] == 1)[0]

            d_assigned = self.select_driver(eligible_drivers, drivers)

            driver = drivers[d_assigned]
            rider = riders[r_selected]

            deviated_path = self.calculate_deviated_path_for_assignment(driver, rider)
            DP_assigned[driver.id]['driver_path'] = deviated_path
            DP_assigned[driver.id]['riders'].append({
                'id': rider.id,
                'source': rider.source,
                'destination': rider.destination
            })

            self.update_eligibility(d_assigned, r_selected, drivers, riders)

        return DP_assigned

    def select_driver(self, eligible_drivers, drivers):
        """Select a driver based on seat availability."""
        if len(eligible_drivers) == 1:
            return eligible_drivers[0]
        else:
            max_seats = -1
            drivers_with_max_seats = []
            for idx in eligible_drivers:
                if drivers[idx].seats > max_seats:
                    max_seats = drivers[idx].seats
                    drivers_with_max_seats = [idx]
                elif drivers[idx].seats == max_seats:
                    drivers_with_max_seats.append(idx)
            return random.choice(drivers_with_max_seats)

    def calculate_deviated_path_for_assignment(self, driver, rider):
        """Calculate the full deviated path for the driver to pick up the rider."""
        path_to_rider_source, _ = self.shortest_path_distance(driver.source, rider.source)
        rider_path, _ = self.shortest_path_distance(rider.source, rider.destination)
        path_from_rider_destination, _ = self.shortest_path_distance(rider.destination, driver.destination)
        return path_to_rider_source + rider_path[1:] + path_from_rider_destination[1:]

    def update_eligibility(self, d_assigned, r_selected, drivers, riders):
        """Update eligibility matrix after assigning a rider to a driver."""
        for rj in range(len(riders)):
            if self.ER[d_assigned][rj] == 1:
                self.ER[d_assigned][rj] = 0

        for di in range(len(drivers)):
            if self.ER[di][r_selected] == 1:
                self.ER[di][r_selected] = 0

        drivers[d_assigned].seats -= 1
        self.update_offers()


class RideSharing:
    def __init__(self, map_obj, driver_file, rider_file):
        self.map_obj = map_obj
        self.drivers = []  # Initialize as an empty list
        self.riders = []   # Initialize as an empty list
        self.load_drivers(driver_file)  # Load drivers from the provided JSON data
        self.load_riders(rider_file)
        self.draw_driver_coordinates()
        self.draw_rider_coordinates()
        self.eligibility_matrix = EligibilityRiderMatrix(self.map_obj)
        self.total_initial_seats = sum(driver.seats for driver in self.drivers)

    def load_drivers(self, driver_file):
        """Load drivers from the JSON file."""
        with open(driver_file, 'r') as f:
            data = json.load(f)
            for driver in data:
                try:
                    # Check if source and destination are valid coordinates
                    if not isinstance(driver['source'], list) or len(driver['source']) != 2:
                        raise ValueError(f"Invalid source coordinates for driver {driver['id']}: {driver['source']}")
                    if not isinstance(driver['destination'], list) or len(driver['destination']) != 2:
                        raise ValueError(f"Invalid destination coordinates for driver {driver['id']}: {driver['destination']}")

                    # Convert source and destination to tuples of floats
                    source = tuple(map(float, driver['source']))
                    destination = tuple(map(float, driver['destination']))

                    # Create the driver object
                    driver_obj = Driver(self.map_obj, driver['id'], source, destination, driver['seats'], driver['threshold'])
                    self.drivers.append(driver_obj)  # Add driver object to the list
                except ValueError as ve:
                    print(f"Error loading driver {driver['id']}: {ve}")
                    
    def load_riders(self, rider_file):
        """Load riders from the JSON file."""
        with open(rider_file, 'r') as f:
            data = json.load(f)
            for rider in data:
                # Convert source and destination to tuples of floats
                source = tuple(map(float, rider['source']))
                destination = tuple(map(float, rider['destination']))
                rider_obj = Rider(self.map_obj, rider['id'], source, destination)
                self.riders.append(rider_obj)  # Add rider object to the list
                
    def draw_driver_coordinates(self, save_path='./Output1/driver_coordinates.png'):
        """Draw the driver's source and destination coordinates on the map's existing axes, save the figure, and annotate with driver IDs."""
        # Get the axes from the existing map plot
        fig, ax = plt.subplots()

        # Draw the existing map graph
        ox.plot_graph(self.map_obj.graph, ax=ax, node_size=10, node_color='red', edge_color='gray', edge_linewidth=0.8, bgcolor='white')

        for driver in self.drivers:
            # Unpack driver's source and destination
            source_lat, source_lon = driver.source
            destination_lat, destination_lon = driver.destination

            # Plot the source as a green point
            ax.plot(source_lon, source_lat, 'go', markersize=5, label='Driver Source' if 'Driver Source' not in ax.get_legend_handles_labels()[1] else "")
            # Plot the destination as a red point
            ax.plot(destination_lon, destination_lat, 'ro', markersize=5, label='Driver Destination' if 'Driver Destination' not in ax.get_legend_handles_labels()[1] else "")

            # Annotate with the driver ID at the source and destination
            ax.text(source_lon, source_lat, f"ID: {driver.id}", fontsize=4, color='black', ha='right', va='bottom')
            ax.text(destination_lon, destination_lat, f"ID: {driver.id}", fontsize=4, color='black', ha='right', va='bottom')

        # Optionally, add labels and title
        ax.set_title('Driver Coordinates on Map')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()

        # Save the figure
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Driver coordinates plot saved successfully at: {save_path}")
        
    def draw_rider_coordinates(self, save_path='./Output1/rider_coordinates.png'):
        """Draw the rider's source and destination coordinates on the map's existing axes, save the figure, and annotate with rider IDs."""
        # Get the axes from the existing map plot
        fig, ax = plt.subplots()

        # Draw the existing map graph
        ox.plot_graph(self.map_obj.graph, ax=ax, node_size=10, node_color='red', edge_color='gray', edge_linewidth=0.8, bgcolor='white')

        for rider in self.riders:
            # Unpack rider's source and destination
            source_lat, source_lon = rider.source
            destination_lat, destination_lon = rider.destination

            # Plot the source as a green point
            ax.plot(source_lon, source_lat, 'go', markersize=5, label='Rider Source' if 'Rider Source' not in ax.get_legend_handles_labels()[1] else "")
            # Plot the destination as a red point
            ax.plot(destination_lon, destination_lat, 'ro', markersize=5, label='Rider Destination' if 'Rider Destination' not in ax.get_legend_handles_labels()[1] else "")

            # Annotate with the rider ID at the source and destination
            ax.text(source_lon, source_lat, f"ID: {rider.id}", fontsize=4, color='black', ha='right', va='bottom')
            ax.text(destination_lon, destination_lat, f"ID: {rider.id}", fontsize=4, color='black', ha='right', va='bottom')

        # Optionally, add labels and title
        ax.set_title('Rider Coordinates on Map')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()

        # Save the figure
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rider coordinates plot saved successfully at: {save_path}")
                
    def assign_riders(self):
        """Assign riders to drivers."""
        self.eligibility_matrix.calculate(self.drivers, self.riders)
        assigned_riders = self.eligibility_matrix.assign_riders_to_drivers(self.drivers, self.riders)
        return assigned_riders

    def run(self):
        """Main function to run the ride-sharing assignment."""
        assigned_riders = self.assign_riders()
        # self.plot_routes(assigned_riders)
        self.output_assignment_summary(assigned_riders)

    def output_assignment_summary(self, assigned_riders):
        """Output the assignment summary to a text file."""
        with open('./Output1/output.txt', 'w') as f:
            for driver in self.drivers:
                f.write(f"Driver {driver.id} with route {driver.source} to {driver.destination}:\n")
                if driver.id in assigned_riders:
                    for rider in assigned_riders[driver.id]['riders']:
                        f.write(f"\tAssigned Rider {rider['id']} from {rider['source']} to {rider['destination']}\n")
                else:
                    f.write("\tNo riders assigned\n")
                f.write(f"\tSeats remaining: {driver.seats}\n\n")


if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set path for the map file
    map_dir = os.path.join(script_dir, 'map')
    # Load the OSM graph data from the GraphML file
    graph_file = os.path.join(map_dir, 'graph.graphml')

    # Set paths for the input files (drivers and riders)
    input_dir = os.path.join(script_dir, 'Input1')
    driver_file = os.path.join(input_dir, 'drivers.json')
    rider_file = os.path.join(input_dir, 'riders.json')
 
    # Create the map object with the loaded graph
    city_map = Map(graph_file)  # pois_file removed

    # Create the ride-sharing object
    ride_sharing = RideSharing(city_map, driver_file, rider_file)

    # Redirect output to a text file (e.g., for debugging purposes)
    sys.stdout = open('./Output1/output.txt', 'w')

    # Run the ride-sharing system to assign riders to drivers
    ride_sharing.run()
    
    # Optional: Close the file output stream after running
    sys.stdout.close()