import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Function to read the CSV file and return a list of edges (node1, node2, weight)
def read_graph_from_csv(file_path):
    edges = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if it exists
        for row in reader:
            node1, node2, weight = int(row[0]), int(row[1]), int(float(row[2]))  # Convert weight to integer
            edges.append((node1, node2, weight))
    return edges

# Function to read the driver CSV file and return a list of drivers
def read_driver_info(file_path):
    drivers = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)  # Use DictReader to map column names to values
        for row in reader:
            # Construct a dictionary for each driver with appropriate type conversions
            driver = {
                'id': row['id'],  # Driver ID remains a string
                'source': int(row['source']),  # Source converted to integer
                'destination': int(row['destination']),  # Destination converted to integer
                'seats': int(row['seats']),  # Seats converted to integer
                'threshold': int(row['threshold'])  # Threshold converted to integer
            }
            drivers.append(driver)
    return drivers

# Function to read the rider CSV file and return a list of riders
def read_rider_info(file_path):
    riders = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)  # Use DictReader to map column names to values
        for row in reader:
            # Construct a dictionary for each rider with appropriate type conversions
            rider = {
                'id': row['id'],  # Rider ID remains a string
                'source': int(row['source']),  # Source converted to integer
                'destination': int(row['destination'])  # Destination converted to integer
            }
            riders.append(rider)
    return riders

# Function to find the shortest path between two nodes
def shortest_path_distance(graph, source, target):
    try:
        # Using Dijkstra's algorithm for weighted shortest path
        path = nx.shortest_path(graph, source=source, target=target, weight='weight')
        path_length = nx.shortest_path_length(graph, source=source, target=target, weight='weight')
        return path, path_length  # Return both path and path length
    except nx.NetworkXNoPath:
        return None, float('inf')  # Return None for path if no path exists

# Function to calculate the Eligibility Rider Matrix (ER)
def calculate_eligibility_rider_matrix(graph, drivers, riders):
    num_drivers = len(drivers)
    num_riders = len(riders)

    # Step 1: Initialize the ER matrix
    ER = np.zeros((num_drivers, num_riders), dtype=int)

    # Step 2: Loop through each driver
    for i in range(num_drivers):
        driver = drivers[i]
        SP, sp_length = shortest_path_distance(graph, driver['source'], driver['destination'])  # Get both path and path length
        t = driver['threshold']  # Threshold as a time factor
        # print(SP)
        # Step 3: Compute maximum distance M Pi
        MP = sp_length * (1 + (t / 100))

        # Step 4: Loop through each rider
        for j in range(num_riders):
            rider = riders[j]

            # Step 5: Calculate Deviated Path Distance (DP)
            DP1, dp1_length = shortest_path_distance(graph, driver['source'], rider['source'])
            DP2, dp2_length = shortest_path_distance(graph, rider['source'], rider['destination'])
            DP3, dp3_length = shortest_path_distance(graph, rider['destination'], driver['destination'])
          
            # Sum the lengths, if any path is None, set DP to infinity
            DP = (dp1_length if dp1_length != float('inf') else float('inf')) + \
                 (dp2_length if dp2_length != float('inf') else float('inf')) + \
                 (dp3_length if dp3_length != float('inf') else float('inf'))

            # Step 6: Check eligibility
            if DP <= MP:
                # print(driver['id'],rider['id'])
                # print(DP,MP)
                # print(f'{DP1},{DP2},{DP3}')
                ER[i][j] = 1  # Rider is eligible for driver i
         
    # Step 11: Initialize the Offer array
    offers = np.zeros(num_riders, dtype=int)

    # Step 13: Count offers for each rider
    for r in range(num_riders):
        for d in range(num_drivers):
            offers[r] += ER[d][r]  # Count drivers who can offer a ride to rider r

    return ER, offers

def assign_riders_to_drivers(ER, offers, drivers, riders, graph):
    DP_assigned = {driver['id']: {'driver_path': [], 'riders': []} for driver in drivers}  # Store paths for each driver

    # Repeat while there are any offers available
    while np.sum(offers) > 0:
        # Step 1: Identify Riders with Minimum Offers
        non_zero_offers = offers[offers > 0]  # Filter out zero offers
        if non_zero_offers.size == 0:
            break  # Exit if there are no non-zero offers

        Min_offer = np.min(non_zero_offers)  # Minimum non-zero offer value
        Min_offer_set = np.where(offers == Min_offer)[0]  # Indices of riders with minimum offers

        # Step 2: Select a Rider
        if len(Min_offer_set) == 1:
            r_selected = Min_offer_set[0]  # Only one rider selected
        else:
            r_selected = random.choice(Min_offer_set)  # Randomly select one rider

        # Step 3: Assign the Selected Rider to a Driver
        eligible_drivers = np.where(ER[:, r_selected] == 1)[0]  # Identify eligible drivers

        if len(eligible_drivers) == 1:
            d_assigned = eligible_drivers[0]  # Only one driver available
        else:
            # Select the driver with the maximum number of available seats
            max_seats = -1
            drivers_with_max_seats = []

            for driver_idx in eligible_drivers:
                if drivers[driver_idx]['seats'] > max_seats:
                    max_seats = drivers[driver_idx]['seats']
                    drivers_with_max_seats = [driver_idx]
                elif drivers[driver_idx]['seats'] == max_seats:
                    drivers_with_max_seats.append(driver_idx)

            # If there is a tie, randomly choose one driver
            d_assigned = random.choice(drivers_with_max_seats)

        # Step 4: Calculate the Path for the Driver and Rider
        driver = drivers[d_assigned]  # Use the driver index to get driver object
        rider = riders[r_selected]    # Use the rider index to get rider object
        
        # Get the deviated path including the rider
        path_to_rider_source, _ = shortest_path_distance(graph, driver['source'], rider['source'])
        rider_path, _ = shortest_path_distance(graph, rider['source'], rider['destination'])
        path_from_rider_destination, _ = shortest_path_distance(graph, rider['destination'], driver['destination'])
        # Combine all paths into one deviated path
        deviated_path = path_to_rider_source + rider_path[1:] + path_from_rider_destination[1:]  # Avoid duplicating nodes
        # Add the deviated path to the assigned driver’s record
        DP_assigned[driver['id']]['driver_path'] = deviated_path
        DP_assigned[driver['id']]['riders'].append({
            'rider_id': rider['id'],
            'rider_path': rider_path,
            'source': rider['source'],
            'destination': rider['destination']
        })
        # Step 5: Handle Deviated Drivers
        for rj in range(len(offers)):
            if ER[d_assigned][rj] == 1:
                # Check if rider rj's route does not lie on the deviated route
                if not is_on_deviated_route(driver['id'], riders[rj], DP_assigned):
                    ER[d_assigned][rj] = 0  # Set eligibility to 0 for rj
        
       
        # Set eligibility for the assigned rider to 0 for all drivers (rider is assigned)
        for d in range(len(drivers)):
            ER[d][r_selected] = 0  # Set eligibility to 0 for assigned rider

        # Decrease the seat count for the assigned driver
        drivers[d_assigned]['seats'] -= 1

        # Step 6: Handle Seat Depletion
        if drivers[d_assigned]['seats'] == 0:
            # Set eligibility to 0 for all riders (this driver can no longer accept any riders)
            ER[d_assigned] = np.zeros(len(riders))

        # Step 7: Update Offer Array
        offers = np.sum(ER, axis=0)  # Update offers based on remaining eligibility
        print(offers)

    return DP_assigned

def is_on_deviated_route(driver_id, rider, DP_assigned):
    # Retrieve the deviated path from DP_assigned for the specific driver
    driver_path = DP_assigned[driver_id]['driver_path']

    # Check if both the rider's source and destination lie on the driver's deviated path
    if rider['source'] in driver_path and rider['destination'] in driver_path:
        return True
    return False

# Function to draw the graph with driver and rider locations
def draw_graph_with_positions(graph, drivers, riders, filename):
    plt.figure(figsize=(10, 8))

    # Get positions for the nodes
    pos = nx.spring_layout(graph)  # Positions for all nodes

    # Draw the graph
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', edge_color='gray')

    # Draw driver nodes
    for driver in drivers:
        plt.annotate(f"Driver {driver['id']} (S: {driver['source']}, D: {driver['destination']})", 
                     xy=pos[driver['source']], 
                     xytext=(5, 5), 
                     textcoords='offset points', 
                     fontsize=9, 
                     color='black')

    # Draw rider nodes
    for rider in riders:
        plt.annotate(f"Rider {rider['id']} (S: {rider['source']}, D: {rider['destination']})", 
                     xy=pos[rider['source']], 
                     xytext=(5, -15), 
                     textcoords='offset points', 
                     fontsize=9, 
                     color='red')

    plt.title('Driver and Rider Positions')
    plt.savefig(filename)  # Save the figure to a file

# Main function to create the graph and find the shortest path
def main():
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'Input')
    
    # Set paths for input files
    graph_file = os.path.join(input_dir, 'Graph.csv')
    driver_file = os.path.join(input_dir, 'Driver.csv')
    rider_file = os.path.join(input_dir, 'Rider.csv')
    
    # Ensure the Output directory exists
    output_dir = os.path.join(script_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV data and add edges to the graph
    edges = read_graph_from_csv(graph_file)
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    # Drivers Information
    drivers = read_driver_info(driver_file)

    # Riders Information
    riders = read_rider_info(rider_file)
    
    # Graph information
    graph_draw_file = os.path.join(output_dir, 'driver_rider_graph.png')  # Update path
    
    # Calculate the total seats
    total_seats = sum(driver['seats'] for driver in drivers)

    # Calculate the total number of riders
    total_riders = len(riders)
    
    # Draw the graph with driver and rider positions
    draw_graph_with_positions(G, drivers, riders, graph_draw_file)
    
    # Calculate the Eligibility Rider Matrix and Offers
    ER, offers = calculate_eligibility_rider_matrix(G, drivers, riders)

    # Write the output to a file
    output_file_path = os.path.join(output_dir, 'Output.txt')  # Update path
    with open(output_file_path, 'w') as output_file:
        # Summarize the input data
        output_file.write("===== Input Data =====\n")

        output_file.write("\nDrivers Information:\n")
        for driver in drivers:
            output_file.write(f"Driver ID: {driver['id']}, Source: {driver['source']}, "
                              f"Destination: {driver['destination']}, Seats: {driver['seats']}, "
                              f"Threshold: {driver['threshold']}\n")

        output_file.write("\nRiders Information:\n")
        for rider in riders:
            output_file.write(f"Rider ID: {rider['id']}, Source: {rider['source']}, "
                              f"Destination: {rider['destination']}\n")

        output_file.write("\n===== Offer Count for Each Rider =====\n")
        for r, offer_count in enumerate(offers):
            output_file.write(f"Rider {r + 1} has {offer_count} offers\n")
            
        # Assign riders to drivers
        DPassigned = assign_riders_to_drivers(ER, offers, drivers, riders, G)

        # Write the assigned riders, their paths, and remaining seats for each driver
        output_file.write("\n===== Assigned Riders, Paths, and Remaining Seats for Each Driver =====\n")
        for driver_id, assignment in DPassigned.items():
            driver_data = next(driver for driver in drivers if driver['id'] == driver_id)
            output_file.write(f"Driver {driver_id}:\n")
            output_file.write(f"  Driver Path: {assignment['driver_path']}\n")
            output_file.write(f"  Remaining Seats: {driver_data['seats']}\n")
            for rider in assignment['riders']:
                output_file.write(f"  Rider {rider['rider_id']} assigned on path {rider['rider_path']} "
                                  f"from {rider['source']} to {rider['destination']}\n")
        
        # Write the total seats, total riders, and total number of accommodated riders
        output_file.write("\n===== Summary =====\n")
        output_file.write(f"Total Seats: {total_seats}\n")
        output_file.write(f"Total Riders: {total_riders}\n")
        
        # Calculate the total accommodated riders
        total_accommodated_riders = sum(len(assignment['riders']) for assignment in DPassigned.values())
        output_file.write(f"Total Number of Accommodated Riders: {total_accommodated_riders} out of {total_riders}\n")


# Check if this script is being run directly
if __name__ == "__main__":
    main()
