import evaluateShared
import networkx as nx
import numpy as np
import sys

ORIGIN = evaluateShared.Point(0, 0)
MAX_SHIFT_TIME = 12 * 60  # 12 hour shift * 60 minutes
DRIVER_PENALTY = 500


def create_graph(vrp: evaluateShared.VRP) -> nx.DiGraph:
    """Creates a graph representing the VRP"""
    graph = nx.DiGraph()
    for load in vrp.loads:
        # Origin directs to all pickup nodes
        graph.add_edge("0", load.id + "a", weight=evaluateShared.distanceBetweenPoints(ORIGIN, load.pickup))

        # Pickup nodes direct to their respective dropoff nodes
        graph.add_edge(load.id + "a", load.id + "b",
                       weight=evaluateShared.distanceBetweenPoints(load.pickup, load.dropoff))

        # Dropoff nodes also direct to origin
        graph.add_edge(load.id + "b", "0", weight=evaluateShared.distanceBetweenPoints(load.dropoff, ORIGIN))

        # Dropoff nodes direct to any pickup nodes, except their own pickup node
        # ASSUMPTION: ids are sequential from 1 to the number of nodes, and they are added into loads sequentially
        for load_id in range(1, len(vrp.loads) + 1):
            if str(load_id) != load.id:
                graph.add_edge(load.id + "b", str(load_id) + "a",
                               weight=evaluateShared.distanceBetweenPoints(load.dropoff, vrp.loads[load_id - 1].pickup))

    return graph


def determine_driver_indices_and_cost(loads: [evaluateShared.Load], chromosome: [int]) -> [int]:
    """Determine where to insert new drivers based on shift time and count up the cost"""
    drivers = [0]  # each element indicates the index of the chromosome where a new driver takes over
    cur_load = loads[chromosome[0] - 1]

    origin_to_pickup_time = evaluateShared.distanceBetweenPoints(ORIGIN, cur_load.pickup)
    cur_load_time = evaluateShared.distanceBetweenPoints(cur_load.pickup, cur_load.dropoff)
    cur_dropoff_to_origin_time = evaluateShared.distanceBetweenPoints(cur_load.dropoff, ORIGIN)
    shift_time = origin_to_pickup_time + cur_load_time
    cost = shift_time
    for j in range(1, len(chromosome)):
        next_load = loads[chromosome[j] - 1]

        cur_to_next_load_time = evaluateShared.distanceBetweenPoints(cur_load.dropoff, next_load.pickup)
        next_load_time = evaluateShared.distanceBetweenPoints(next_load.pickup, next_load.dropoff)
        next_dropoff_to_origin_time = evaluateShared.distanceBetweenPoints(next_load.dropoff, ORIGIN)

        # With one load, we are given the guarantee: load_time + dropoff_to_origin_time <= MAX_SHIFT_TIME
        if shift_time + cur_to_next_load_time + next_load_time + next_dropoff_to_origin_time > MAX_SHIFT_TIME:
            origin_to_pickup_time = evaluateShared.distanceBetweenPoints(ORIGIN, next_load.pickup)

            cost += cur_dropoff_to_origin_time + DRIVER_PENALTY + origin_to_pickup_time + next_load_time
            shift_time = origin_to_pickup_time + next_load_time

            drivers.append(j)
        else:
            shift_time += cur_to_next_load_time + next_load_time
            cost += cur_to_next_load_time + next_load_time

        cur_load = next_load
        cur_dropoff_to_origin_time = next_dropoff_to_origin_time

    return cost, drivers


def mutate_chromosome(chromosome: [int], num_mutations: int, num_loads: int) -> [int]:
    mutation_random = np.random.default_rng()
    for i in range(num_mutations):
        idx_1 = int(mutation_random.random() * num_loads)
        idx_2 = int(mutation_random.random() * num_loads)

        temp = chromosome[idx_1]
        chromosome[idx_1] = chromosome[idx_2]
        chromosome[idx_2] = temp
    return chromosome


def genetic_algorithm(loads: [evaluateShared.Load],
                      population_size: int,
                      max_mutation_proportion: float,
                      max_num_generations: int,
                      performance_change_threshold: int) -> []:
    """Performs the genetic algorithm to solve the VRP"""
    # Set up random generator instance
    genetic_random = np.random.default_rng()

    # Initialize random population
    chromosome = np.arange(1, len(loads) + 1)  # ASSUMPTION: load ids are sequential from 1
    chromosomes = []
    costs = []
    driver_lists = []  # each element indicates the load index where the driver takes over
    for i in range(population_size):
        genetic_random.shuffle(chromosome)
        chromosomes.append(chromosome.copy())

        cost, drivers = determine_driver_indices_and_cost(loads, chromosome)
        costs.append(cost)
        driver_lists.append(drivers)

    # Mutate over many generations
    prev_min_cost = costs[0]
    for generation_num in range(max_num_generations):
        # Mutate chromosomes based on relative cost
        costs = np.array(costs)
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        new_chromosomes = []
        for i in range(population_size):
            max_num_mutations = population_size * max_mutation_proportion
            num_mutations = int((costs[i] - min_cost) / (max_cost - min_cost) * max_num_mutations)
            new_chromosomes.append(mutate_chromosome(chromosomes[i].copy(), num_mutations, len(loads)))

        chromosomes = new_chromosomes
        costs = []
        driver_lists = []
        for i in range(population_size):
            cost, drivers = determine_driver_indices_and_cost(loads, chromosomes[i])
            costs.append(cost)
            driver_lists.append(drivers)

        min_cost = np.min(np.array(costs))
        if abs(min_cost - prev_min_cost) < performance_change_threshold:
            break

        prev_min_cost = min_cost

    # Find min cost chromosome
    min_cost = np.min(np.array(costs))
    min_cost_idx = costs.index(min_cost)
    return chromosomes[min_cost_idx], driver_lists[min_cost_idx]


def solve_vrp() -> None:
    """Main function for solving the vehicle routing system"""
    # Set up data for processing
    vrp = evaluateShared.loadProblemFromFile(sys.argv[1])

    # Solve VRP using genetic algorithm
    load_order, drivers = genetic_algorithm(vrp.loads,
                                            population_size=int(0.2 * len(vrp.loads)) + 20,
                                            max_mutation_proportion=0.2,
                                            max_num_generations=200,
                                            performance_change_threshold=0) # 0 disables this feature

    # Print formatted solution
    schedules = [''] * len(drivers)
    for i in range(len(drivers) - 1):
        schedules[i] = str(load_order[drivers[i]: drivers[i + 1]].tolist())
    schedules[len(drivers) - 1] = str(load_order[drivers[len(drivers) - 1]:].tolist())
    print('\n'.join(schedules))


if __name__ == "__main__":
    solve_vrp()
