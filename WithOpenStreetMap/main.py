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
    def __init__(self, graph_file, pois_file):
        self.graph = self.load_osm_data(graph_file)
        self.pois = self.load_pois_data(pois_file)

    @staticmethod
    def load_osm_data(graph_file):
        """Load the street network from a local GraphML file."""
        try:
            print("Loading graph data...")
            return ox.load_graphml(graph_file)
        except Exception as e:
            print(f"Error loading OSM data: {e}")
            raise

    @staticmethod
    def load_pois_data(pois_file):
        """Load POIs from a local GeoJSON file."""
        try:
            print("Loading POIs data...")
            return gpd.read_file(pois_file)
        except Exception as e:
            print(f"Error loading POIs data: {e}")
            raise

    def find_location(self, location_name):
        """Search for a named location in the downloaded OSM data (POIs)."""
        print(f"Finding location for: {location_name}")
        matching_places = self.pois[self.pois["name"].str.contains(location_name, case=False, na=False)]
        if not matching_places.empty:
            location_point = matching_places.iloc[0].geometry.centroid
            return (location_point.y, location_point.x)  # Return latitude, longitude
        else:
            raise ValueError(f"Could not find coordinates for location: {location_name} in the OSM data.")

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
    def __init__(self, map_obj, driver_id, source, destination, seats, threshold):
        self.map_obj = map_obj
        self.driver_id = driver_id  # Added driver ID
        self.source_name = source
        self.destination_name = destination
        self.source = self.map_obj.find_location(source)
        self.destination = self.map_obj.find_location(destination)
        self.seats = seats
        self.threshold = threshold
        self.route = None

    def draw_route(self, ax, color='b'):
        """Draw the driver's route on the provided axes with the specified color."""
        route = self.map_obj.draw_route(self.source, self.destination)

        if route:  # Only plot if there is a valid route
            print(f"Drawing driver route from {self.source_name} to {self.destination_name}.")

            route_lines = [(self.map_obj.graph.nodes[n1]['y'], self.map_obj.graph.nodes[n1]['x']) for n1 in route]

            x, y = zip(*route_lines)  # Unzip into x and y coordinates

            ax.plot(y, x, color=color, linewidth=4)  # Draw the route in the specified color


class Rider:
    def __init__(self, map_obj, rider_id, source, destination):
        self.map_obj = map_obj
        self.rider_id = rider_id  # Added rider ID
        self.source_name = source
        self.destination_name = destination
        self.source = self.map_obj.find_location(source)
        self.destination = self.map_obj.find_location(destination)
        self.matched_driver = None

    def draw_route(self, ax, color='g'):
        """Draw the rider's route on the provided axes with the specified color."""
        route = self.map_obj.draw_route(self.source, self.destination)

        if route:  # Only plot if there is a valid route
            print(f"Drawing rider route from {self.source_name} to {self.destination_name}.")

            route_lines = [(self.map_obj.graph.nodes[n1]['y'], self.map_obj.graph.nodes[n1]['x']) for n1 in route]

            x, y = zip(*route_lines)  # Unzip into x and y coordinates

            ax.plot(y, x, color=color, linewidth=4)  # Draw the route in the specified color


class EligibilityRiderMatrix:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager
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

            for j, rider in enumerate(riders):
                DP = self.calculate_deviated_path(driver, rider)
                if DP <= MP:
                    self.ER[i][j] = 1

        self.update_offers()
    
    def shortest_path_distance(self, source, target):
        try:
            path = nx.shortest_path(self.graph_manager.graph, source=source, target=target, weight='weight')
            path_length = nx.shortest_path_length(self.graph_manager.graph, source=source, target=target, weight='weight')
            return path, path_length
        except nx.NetworkXNoPath:
            return None, float('inf')
        
    def calculate_deviated_path(self, driver, rider):
        DP1, dp1_length = self.shortest_path_distance(driver.source, rider.source)
        DP2, dp2_length = self.shortest_path_distance(rider.source, rider.destination)
        DP3, dp3_length = self.shortest_path_distance(rider.destination, driver.destination)

        DP = (dp1_length if dp1_length != float('inf') else float('inf')) + \
             (dp2_length if dp2_length != float('inf') else float('inf')) + \
             (dp3_length if dp3_length != float('inf') else float('inf'))
        return DP
    
    def update_offers(self):
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
                'rider_id': rider.id,
                'source': rider.source,
                'destination': rider.destination
            })

            self.update_eligibility(d_assigned, r_selected, drivers, riders, DP_assigned)

        return DP_assigned
    
    def select_driver(self, eligible_drivers, drivers):
        if len(eligible_drivers) == 1:
            return eligible_drivers[0]
        else:
            max_seats = -1
            drivers_with_max_seats = []
            for driver_idx in eligible_drivers:
                if drivers[driver_idx].seats > max_seats:
                    max_seats = drivers[driver_idx].seats
                    drivers_with_max_seats = [driver_idx]
                elif drivers[driver_idx].seats == max_seats:
                    drivers_with_max_seats.append(driver_idx)
            return random.choice(drivers_with_max_seats)
        
    def calculate_deviated_path_for_assignment(self, driver, rider):
        path_to_rider_source, _ = self.shortest_path_distance(driver.source, rider.source)
        rider_path, _ = self.shortest_path_distance(rider.source, rider.destination)
        path_from_rider_destination, _ = self.shortest_path_distance(rider.destination, driver.destination)
        return path_to_rider_source + rider_path[1:] + path_from_rider_destination[1:]

    def update_eligibility(self, d_assigned, r_selected, drivers, riders, DP_assigned):
        for rj in range(len(riders)):
            if self.ER[d_assigned][rj] == 1:
                if not self.is_on_deviated_route(drivers[d_assigned].id, riders[rj], DP_assigned):
                    self.ER[d_assigned][rj] = 0

        for d in range(len(drivers)):
            self.ER[d][r_selected] = 0

        drivers[d_assigned].seats -= 1
        if drivers[d_assigned].seats == 0:
            self.ER[d_assigned] = np.zeros(len(riders))

        self.update_offers()

    def is_on_deviated_route(self, driver_id, rider, DP_assigned):
        driver_path = DP_assigned[driver_id]['driver_path']
        
        if rider.source in driver_path and rider.destination in driver_path:
            return True
        
        return False

class RideSharing:
    def __init__(self, graph_file, pois_file):
        self.map = Map(graph_file, pois_file)
        self.drivers = []
        self.riders = []
        self.eligibility_matrix = EligibilityRiderMatrix(self.map)
        self.total_initial_seats = sum(driver.seats for driver in self.drivers)

    def add_driver(self, driver_id, source, destination, seats, threshold):
        driver = Driver(self.map, driver_id, source, destination, seats, threshold)
        self.drivers.append(driver)

    def add_rider(self, rider_id, source, destination):
        rider = Rider(self.map, rider_id, source, destination)
        self.riders.append(rider)

    def load_data_from_json(self, driver_file, rider_file):
        with open(driver_file) as f:
            drivers_data = json.load(f)
            for driver in drivers_data:
                self.add_driver(driver['id'], driver['source'], driver['destination'], driver['seats'], driver['threshold'])

        with open(rider_file) as f:
            riders_data = json.load(f)
            for rider in riders_data:
                self.add_rider(rider['id'], rider['source'], rider['destination'])
                
    def output_results(self, DPassigned):
        total_remaining_seats = sum(driver.seats for driver in self.drivers)
        total_riders = len(self.riders)
        
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'Output')
        
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

        with open(os.path.join(output_dir, 'Output.txt'), 'w') as output_file:
            output_file.write("===== Input Data =====\n")
            output_file.write("\nDrivers Information:\n")
            for driver in self.drivers:
                output_file.write(f"Driver ID: {driver.id}, Source: {driver.source}, "
                                  f"Destination: {driver.destination}, Seats: {driver.seats}, "
                                  f"Threshold: {driver.threshold}\n")

            output_file.write("\nRiders Information:\n")
            for rider in self.riders:
                output_file.write(f"Rider ID: {rider.id}, Source: {rider.source}, "
                                  f"Destination: {rider.destination}\n")

            output_file.write("\n===== Assigned Riders, Paths, and Remaining Seats for Each Driver =====\n")
            for driver_id, assignment in DPassigned.items():
                output_file.write(f"Driver {driver_id}:\n")
                output_file.write(f"  Driver Path: {assignment['driver_path']}\n")
                for rider in assignment['riders']:
                    output_file.write(f"    Rider ID: {rider['rider_id']}, "
                                      f"Source: {rider['source']}, Destination: {rider['destination']}\n")

            output_file.write(f"\nTotal Current Seats Available: {total_remaining_seats}/{self.total_initial_seats}, Total Number of Accommodated Riders: {self.total_initial_seats - total_remaining_seats} out of {total_riders}\n")

        print("Results have been written to Output.txt.")
                
    def run(self):
        self.eligibility_matrix.calculate(self.drivers, self.riders)
        DPassigned = self.eligibility_matrix.assign_riders_to_drivers(self.drivers, self.riders)

        self.output_results(DPassigned)

    def draw_roadmap(self):
        """Draw the roadmap with driver and rider routes."""
        # Set the figure size to make the image larger
        fig, ax = ox.plot_graph(self.map.graph, node_size=0, edge_linewidth=0.8, show=False, close=False, figsize=(15, 15), dpi=100)

        # Plot each driver's route in blue
        for driver in self.drivers:
            driver.draw_route(ax, color='blue')

        # Plot each rider's route in green
        for rider in self.riders:
            rider.draw_route(ax, color='green')

        # Gather driver and rider coordinates for markers
        driver_coords = [driver.source for driver in self.drivers]
        rider_coords = [rider.source for rider in self.riders]

        # Add markers for drivers and riders
        ax.scatter([coord[1] for coord in driver_coords],
                   [coord[0] for coord in driver_coords],
                   c='blue', s=100, label='Driver', alpha=0.7)

        ax.scatter([coord[1] for coord in rider_coords],
                   [coord[0] for coord in rider_coords],
                   c='green', s=100, label='Rider', alpha=0.7)

        # Add legend for drivers and riders
        ax.legend(loc='lower left')

        plt.title("Driver and Rider Routes")
        plt.show()


# Main Program
if __name__ == "__main__":
    # Redirect print statements to output.txt
    output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output', 'output.txt')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Redirect stdout to the output file
    original_stdout = sys.stdout
    with open(output_file_path, 'w') as output_file:
        sys.stdout = output_file

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Set paths for map files
        map_dir = os.path.join(script_dir, 'map')
        # Load map data files
        graph_file = os.path.join(map_dir, 'graph.graphml')
        pois_file = os.path.join(map_dir, 'pois.geojson')

        # Set paths for input files
        input_dir = os.path.join(script_dir, 'Input')
        # Load driver and rider data from JSON files
        driver_file = os.path.join(input_dir, 'drivers.json')
        rider_file = os.path.join(input_dir, 'riders.json')

        ride_sharing = RideSharing(graph_file, pois_file)

        ride_sharing.load_data_from_json(driver_file, rider_file)

        # Draw the roadmap with driver and rider routes
        ride_sharing.draw_roadmap()

    # Restore original stdout
    sys.stdout = original_stdout
    print("Successfully has been written in output.txt")
