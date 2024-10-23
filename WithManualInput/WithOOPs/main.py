import csv
import networkx as nx
import numpy as np
import random
import os

class GraphManager:
    def __init__(self, file_path):
        self.graph = nx.DiGraph()
        self.load_graph(file_path)

    def load_graph(self, file_path):
        edges = self.read_graph_from_csv(file_path)
        for edge in edges:
            self.graph.add_edge(edge[0], edge[1], weight=edge[2])

    @staticmethod
    def read_graph_from_csv(file_path):
        edges = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row if it exists
            for row in reader:
                node1, node2, weight = int(row[0]), int(row[1]), int(float(row[2]))  # Convert weight to integer
                edges.append((node1, node2, weight))
        return edges


class Driver:
    def __init__(self, driver_id, source, destination, seats, threshold):
        self.id = driver_id
        self.source = source
        self.destination = destination
        self.seats = seats
        self.threshold = threshold


class Rider:
    def __init__(self, rider_id, source, destination):
        self.id = rider_id
        self.source = source
        self.destination = destination 


class EligibilityRiderMatrix:
    def __init__(self, graph_manager, output_file):
        self.graph_manager = graph_manager
        self.ER = None
        self.offers = None
        output_file.write("EligibilityRiderMatrix initialized.\n")

    def calculate(self, drivers, riders, output_file):
        num_drivers = len(drivers)
        num_riders = len(riders)

        self.ER = np.zeros((num_drivers, num_riders), dtype=int)
        self.offers = np.zeros(num_riders, dtype=int)
        output_file.write(f"Initialized ER matrix of size {self.ER.shape} and offers: {self.offers}\n")

        for i, driver in enumerate(drivers):
            SP, sp_length = self.shortest_path_distance(driver.source, driver.destination, output_file)
            t = driver.threshold
            MP = sp_length * (1 + (t / 100))
            output_file.write(f"Driver {driver.id}: Shortest Path (SP) length = {sp_length}, Maximum Path (MP) = {MP}\n")

            for j, rider in enumerate(riders):
                DP = self.calculate_deviated_path(driver, rider, output_file)
                output_file.write(f"Rider {rider.id}: Deviated Path (DP) length = {DP}\n")
                if DP <= MP:
                    self.ER[i][j] = 1
                    output_file.write(f"Rider {rider.id} is eligible for Driver {driver.id}\n")

        output_file.write(f"Eligibility matrix after calculation: \n{self.ER}\n")
        self.update_offers(output_file)

    def shortest_path_distance(self, source, target, output_file):
        try:
            path = nx.shortest_path(self.graph_manager.graph, source=source, target=target, weight='weight')
            path_length = nx.shortest_path_length(self.graph_manager.graph, source=source, target=target, weight='weight')
            output_file.write(f"Shortest path from {source} to {target}: {path}, Length: {path_length}\n")
            return path, path_length
        except nx.NetworkXNoPath:
            output_file.write(f"No path found from {source} to {target}.\n")
            return None, float('inf')

    def calculate_deviated_path(self, driver, rider, output_file):
        DP1, dp1_length = self.shortest_path_distance(driver.source, rider.source, output_file)
        DP2, dp2_length = self.shortest_path_distance(rider.source, rider.destination, output_file)
        DP3, dp3_length = self.shortest_path_distance(rider.destination, driver.destination, output_file)

        DP = (dp1_length if dp1_length != float('inf') else float('inf')) + \
             (dp2_length if dp2_length != float('inf') else float('inf')) + \
             (dp3_length if dp3_length != float('inf') else float('inf'))
        output_file.write(f"Deviated path for driver {driver.id} and rider {rider.id}: DP1={dp1_length}, DP2={dp2_length}, DP3={dp3_length}, Total DP={DP}\n")
        return DP

    def update_offers(self, output_file):
        self.offers = np.sum(self.ER, axis=0)
        output_file.write(f"Updated offers: {self.offers}\n")

    def assign_riders_to_drivers(self, drivers, riders, output_file):
        DP_assigned = {driver.id: {'driver_path': [], 'riders': []} for driver in drivers}
        output_file.write(f"Initial DP_assigned: {DP_assigned}\n")

        while np.sum(self.offers) > 0:
            non_zero_offers = self.offers[self.offers > 0]
            output_file.write(f"Non-zero offers: {non_zero_offers}\n")
            if non_zero_offers.size == 0:
                break

            Min_offer = np.min(non_zero_offers)
            Min_offer_set = np.where(self.offers == Min_offer)[0]

            r_selected = Min_offer_set[0] if len(Min_offer_set) == 1 else random.choice(Min_offer_set)
            output_file.write(f"Selected rider {r_selected} with min offer: {Min_offer}\n")
            eligible_drivers = np.where(self.ER[:, r_selected] == 1)[0]
            output_file.write(f"Eligible drivers for rider {r_selected}: {eligible_drivers}\n")

            # Use eligible_drivers to get the driver index
            d_assigned = self.select_driver(eligible_drivers, drivers, output_file)
            output_file.write(f"Assigned driver {d_assigned} to rider {r_selected}\n")

            driver = drivers[d_assigned]  # Assign the driver using the index
            rider = riders[r_selected]

            deviated_path = self.calculate_deviated_path_for_assignment(driver, rider, output_file)
            DP_assigned[driver.id]['driver_path'] = deviated_path
            DP_assigned[driver.id]['riders'].append({
                'rider_id': rider.id,
                'source': rider.source,
                'destination': rider.destination
            })
            output_file.write(f"Updated DP_assigned for driver {driver.id}: {DP_assigned[driver.id]}\n")

            self.update_eligibility(d_assigned, r_selected, drivers, riders, DP_assigned, output_file)

        output_file.write(f"Final DP_assigned: {DP_assigned}\n")
        return DP_assigned


    def select_driver(self, eligible_drivers, drivers, output_file):
        if len(eligible_drivers) == 1:
            output_file.write(f"Only one eligible driver: {eligible_drivers[0]}\n")
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
            selected_driver = random.choice(drivers_with_max_seats)
            output_file.write(f"Selected driver {selected_driver} from drivers with max seats: {drivers_with_max_seats}\n")
            return selected_driver

    def calculate_deviated_path_for_assignment(self, driver, rider, output_file):
        path_to_rider_source, _ = self.shortest_path_distance(driver.source, rider.source, output_file)
        rider_path, _ = self.shortest_path_distance(rider.source, rider.destination, output_file)
        path_from_rider_destination, _ = self.shortest_path_distance(rider.destination, driver.destination, output_file)
        full_path = path_to_rider_source + rider_path[1:] + path_from_rider_destination[1:]
        output_file.write(f"Calculated deviated path for driver {driver.id} and rider {rider.id}: {full_path}\n")
        return full_path

    def update_eligibility(self, d_assigned, r_selected, drivers, riders, DP_assigned, output_file):
        output_file.write(f"Updating eligibility for driver {d_assigned} and rider {r_selected}\n")
        for rj in range(len(riders)):
            if self.ER[d_assigned][rj] == 1:
                if not self.is_on_deviated_route(drivers[d_assigned].id, riders[rj], DP_assigned, output_file):
                    self.ER[d_assigned][rj] = 0
        output_file.write(f"Updated eligibility matrix for driver {d_assigned}: {self.ER[d_assigned]}\n")

        for d in range(len(drivers)):
            self.ER[d][r_selected] = 0

        drivers[d_assigned].seats -= 1
        output_file.write(f"Updated seats for driver {d_assigned}: {drivers[d_assigned].seats}\n")
        if drivers[d_assigned].seats == 0:
            self.ER[d_assigned] = np.zeros(len(riders))

        self.update_offers(output_file)

    def is_on_deviated_route(self, driver_id, rider, DP_assigned, output_file):
        driver_path = DP_assigned[driver_id]['driver_path']
        if rider.source in driver_path and rider.destination in driver_path:
            output_file.write(f"Rider {rider.id} is on the deviated route of driver {driver_id}\n")
            return True
        output_file.write(f"Rider {rider.id} is NOT on the deviated route of driver {driver_id}\n")
        return False


class RideShareSystem:
    def __init__(self, graph_file, driver_file, rider_file, output_file):
        self.graph_manager = GraphManager(graph_file)
        self.drivers = self.load_drivers(driver_file, output_file)
        self.riders = self.load_riders(rider_file, output_file)
        self.eligibility_matrix = EligibilityRiderMatrix(self.graph_manager, output_file)
        self.total_initial_seats = sum(driver.seats for driver in self.drivers)
        output_file.write(f"Total initial seats: {self.total_initial_seats}\n")

    @staticmethod
    def load_drivers(file_path, output_file):
        drivers = []
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                driver = Driver(
                    driver_id=row['id'],
                    source=int(row['source']),
                    destination=int(row['destination']),
                    seats=int(row['seats']),
                    threshold=int(row['threshold'])
                )
                drivers.append(driver)
        output_file.write(f"Loaded drivers: {[driver.__dict__ for driver in drivers]}\n")
        return drivers

    @staticmethod
    def load_riders(file_path, output_file):
        riders = []
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                rider = Rider(
                    rider_id=row['id'],
                    source=int(row['source']),
                    destination=int(row['destination'])
                )
                riders.append(rider)
        output_file.write(f"Loaded riders: {[rider.__dict__ for rider in riders]}\n")
        return riders

    def run(self, output_file):
        output_file.write("Running the ride-sharing system...")
        self.eligibility_matrix.calculate(self.drivers, self.riders, output_file)
        DPassigned = self.eligibility_matrix.assign_riders_to_drivers(self.drivers, self.riders, output_file)
        self.output_results(DPassigned, output_file)

    def output_results(self, DPassigned, output_file):
        total_remaining_seats = sum(driver.seats for driver in self.drivers)
        total_riders = len(self.riders)
        output_file.write(f"Total remaining seats: {total_remaining_seats}, Total riders: {total_riders}\n")
        
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


if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'Input')

    # Set paths for input files
    graph_file = os.path.join(input_dir, 'Graph.csv')
    driver_file = os.path.join(input_dir, 'Driver.csv')
    rider_file = os.path.join(input_dir, 'Rider.csv')
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'Output')
    
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    with open(os.path.join(output_dir, 'Output.txt'), 'w') as output_file:
    
        ride_share_system = RideShareSystem(graph_file, driver_file, rider_file, output_file)
        ride_share_system.run(output_file)
