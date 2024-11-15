# Computational Physics problem: Travelling Salesman Problem (TSP) using Parallel Tempering
# Jaime Raynaud Sanchez
# 2024-10-23
# Data obtained from: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/index.html

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import os


# Function to parse the TSP file and retrieve coordinates and edge weight type
def parse_tsp_file(filename):
    coordinates = []
    edge_weight_type = None
    with open(filename, 'r') as f:
        lines = f.readlines()
        node_section = False
        for line in lines:
            if "EDGE_WEIGHT_TYPE" in line:
                edge_weight_type = line.split(":")[1].strip()
            if "NODE_COORD_SECTION" in line:
                node_section = True
                continue
            if "EOF" in line:
                break
            if node_section:
                parts = line.split()
                index, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                coordinates.append((x, y))
    return coordinates, edge_weight_type

# Function to parse the optimal tour file
def parse_tour_file(filename):
    tour = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        tour_section = False
        for line in lines:
            if "TOUR_SECTION" in line:
                tour_section = True
                continue
            if "-1" in line:  # End of tour section
                break
            if tour_section:
                tour.append(int(line.strip()) - 1)  # Subtract 1 to convert to 0-based index
    return tour

# ATT distance calculation (pseudo-Euclidean)
def att_distance(x1, y1, x2, y2):
    rij = math.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2) / 10.0)
    tij = int(round(rij))
    return tij if tij >= rij else tij + 1

# Euclidean distance calculation (EUC_2D)
def euc_2d_distance(x1, y1, x2, y2):
    return round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

# Function to create distance matrix based on the distance function
def create_distance_matrix(coordinates, edge_weight_type):
    num_cities = len(coordinates)
    distances = np.zeros((num_cities, num_cities))
    
    # Choose the distance function based on EDGE_WEIGHT_TYPE
    if edge_weight_type == "ATT":
        distance_function = att_distance
    elif edge_weight_type == "EUC_2D":
        distance_function = euc_2d_distance
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")

    # Fill the distance matrix
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = distance_function(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1])
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

# Objective function to calculate total distance
def calculate_distance(tour, distances):
    return sum(distances[tour[i], tour[i+1]] for i in range(len(tour)-1)) + distances[tour[-1], tour[0]]

# 2-opt swap to generate neighboring solutions
def two_opt_swap(tour):
    new_tour = tour[:]
    i, j = sorted(random.sample(range(len(tour)), 2))
    new_tour[i:j+1] = reversed(new_tour[i:j+1])
    return new_tour

# Function to compute the total length of a given tour
def compute_tour_length(tour, distances):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distances[tour[i], tour[i + 1]]
    total_distance += distances[tour[-1], tour[0]]  # Return to the starting city
    return total_distance

# Function to compute the optimality gap
def compute_optimality_gap(algorithm_solution, optimal_solution):
    gap_percentage = float(abs((algorithm_solution - optimal_solution) / optimal_solution) * 100)
    return gap_percentage

def compute_avg_nn_distance(distances):
    num_cities = len(distances)
    total_nn_distance = 0
    
    for i in range(num_cities):
        nn_distance = np.min([distances[i, j] for j in range(num_cities) if i != j])
        total_nn_distance += nn_distance
    
    avg_nn_distance = total_nn_distance / num_cities
    return avg_nn_distance

def parallel_tempering(distances, min_t=0.002, max_t=20, num_temps=10, num_iters=10000):
    num_cities = len(distances)
    
    # Create a linearly spaced array between 0 and 1
    linear_space = np.linspace(0, 1, num_temps)
    # Use an exponential transformation to cluster values toward the lower end
    # Adjust the exponent value to control clustering
    exponent = 3
    biased_space = linear_space**exponent
    # Scale the biased values to fit between min_t and max_t
    temperatures = min_t + (max_t - min_t) * biased_space
    for temp in temperatures:
        print(f"{temp:.3f}")

    tours = [random.sample(range(num_cities), num_cities) for _ in range(num_temps)]
    best_solution = min(tours, key=lambda tour: calculate_distance(tour, distances))
    
    length_sums = np.zeros(num_temps)
    variance_sums = 0
    acceptance_rate_sums = np.zeros(num_temps-1)
    
    for iteration in range(num_iters):
        iteration_lengths = np.zeros(num_temps)
        
        for t in range(num_temps):
            current_tour = tours[t]
            new_tour = two_opt_swap(current_tour)
            
            current_distance = calculate_distance(current_tour, distances)
            new_distance = calculate_distance(new_tour, distances)
            
            # Accept with Metropolis criterion
            if new_distance < current_distance or np.exp((current_distance - new_distance) / temperatures[t]) > random.random():
                tours[t] = new_tour
                current_distance = new_distance
                # acceptance_rate_sums[t] += 1

            # Update best solution
            if calculate_distance(new_tour, distances) < calculate_distance(best_solution, distances):
                best_solution = new_tour
            
            # Update length sums
            length_sums[t] += current_distance
            iteration_lengths[t] = current_distance
        
        # Calculate variance for this iteration
        iteration_variance = np.max(iteration_lengths) - np.min(iteration_lengths)
        variance_sums += iteration_variance
        
        # Swap solutions between chains based on temperature
        for t in range(num_temps - 1):
            swap_prob = np.exp((calculate_distance(tours[t], distances) - calculate_distance(tours[t+1], distances)) * (1/temperatures[t] - 1/temperatures[t+1]))
            if swap_prob > random.random():
                tours[t], tours[t+1] = tours[t+1], tours[t]
                acceptance_rate_sums[t] += 1

    # Calculate average lengths, variances, and acceptance rates
    avg_lengths = length_sums / num_iters
    avg_variance = variance_sums / num_iters
    avg_acceptance_rates = acceptance_rate_sums / num_iters
    
    return best_solution, avg_lengths, avg_variance, avg_acceptance_rates, temperatures

def visualize_results(num_cities, optimality_gaps, avg_lengths_list, avg_variances_list, avg_acceptance_rates_list, temperatures_list):
    # Ensure the images directory exists
    if not os.path.exists('images'):
        os.makedirs('images')

    # Plot optimality gap vs number of cities
    plt.figure(figsize=(10, 6))
    plt.plot(num_cities, optimality_gaps, marker='o')
    plt.title('Optimality Gap vs Number of Cities')
    plt.xlabel('Number of Cities')
    plt.ylabel('Optimality Gap (%)')
    plt.savefig('images/optimality_gap_vs_num_cities.png')
    plt.show()

    # Plot average variance vs number of cities
    plt.figure(figsize=(10, 6))
    plt.plot(num_cities, avg_variances_list, marker='o')
    plt.title('Average Variance vs Number of Cities')
    plt.xlabel('Number of Cities')
    plt.ylabel('Average Variance')
    plt.savefig('images/avg_variance_vs_num_cities.png')
    plt.show()

    # Plot average length vs temperature for each city
    fig, axs = plt.subplots(1, len(num_cities), figsize=(25, 5))
    for i in range(len(num_cities)):
        axs[i].plot(temperatures_list[i], avg_lengths_list[i], marker='o')
        axs[i].set_title(f'Average Length vs Temperature\nfor {num_cities[i]} Cities')
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('Average Length')
    plt.tight_layout()
    plt.savefig('images/avg_length_vs_temperature.png')
    plt.show()

    # Plot acceptance rate vs temperature for each city
    fig, axs = plt.subplots(1, len(num_cities), figsize=(25, 5))
    for i in range(len(num_cities)):
        axs[i].plot(temperatures_list[i][:-1], avg_acceptance_rates_list[i], marker='o')
        axs[i].set_title(f'Acceptance Rate vs Temperature\nfor {num_cities[i]} Cities')
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('Acceptance Rate')
    plt.tight_layout()
    plt.savefig('images/acceptance_rate_vs_temperature.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    import os
    
    tsp_folder = 'data/tsp'
    optimal_tour_folder = 'data/optimaltour'

    num_cities_list = []
    optimality_gaps = []
    avg_lengths_list = []
    avg_variances_list = []
    avg_acceptance_rates_list = []
    temperatures_list = []

    tsp_files = [
        'att48.tsp',
        'berlin52.tsp',
        'ch130.tsp',
        'a280.tsp',
        'pcb442.tsp']
    
    for tsp_file in tsp_files:
        if tsp_file.endswith('.tsp'):
            tsp_path = os.path.join(tsp_folder, tsp_file)
            optimal_tour_file = tsp_file.replace('.tsp', '.opt.tour')
            optimal_tour_path = os.path.join(optimal_tour_folder, optimal_tour_file)

            print(f"\n{'=' * 50}")
            print(f"Processing TSP file: {tsp_file}")
            
            # Parse TSP file and create distance matrix
            coordinates, edge_weight_type = parse_tsp_file(tsp_path)
            distances = create_distance_matrix(coordinates, edge_weight_type)

            # Run parallel tempering
            best_solution, avg_lengths, avg_variance, avg_acceptance_rates, temperatures = \
                parallel_tempering(distances, min_t=0.002, max_t=len(distances)*3, num_temps=50, num_iters=100000) #len(distances)*3
            # Compute the optimal tour length and the length found by the algorithm
            optimal_tour = parse_tour_file(optimal_tour_path)
            optimal_tour_length = compute_tour_length(optimal_tour, distances)
            parallel_tempering_tour_length = compute_tour_length(best_solution, distances)

            optimality_gap = compute_optimality_gap(parallel_tempering_tour_length, optimal_tour_length)
            num_cities = len(coordinates)

            num_cities_list.append(num_cities)
            optimality_gaps.append(optimality_gap)
            avg_lengths_list.append(avg_lengths)
            avg_variances_list.append(avg_variance)
            avg_acceptance_rates_list.append(avg_acceptance_rates)
            temperatures_list.append(temperatures)

            print(f"Optimal tour length: {optimal_tour_length}")
            print(f"Parallel tempering tour length: {parallel_tempering_tour_length}")
            print(f"Optimality gap: {optimality_gap:.2f}%")
            print(f"Average lengths: {avg_lengths}")
            print(f"Average variance: {avg_variance}")
            print(f"Average acceptance rates: {avg_acceptance_rates}")
            print(f"{'=' * 50}\n")

    # Visualize results
    visualize_results(num_cities_list, optimality_gaps, avg_lengths_list, avg_variances_list, avg_acceptance_rates_list, temperatures_list)