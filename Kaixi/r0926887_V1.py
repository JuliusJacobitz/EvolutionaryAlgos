import Reporter
import numpy as np
import random


# Replace with your own student number
class r0926887:
    def __init__(self):
        # Initialize the reporter for logging progress
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def length(self, tour, distance_matrix):
        # Calculate the total length of a TSP tour (cycle)
        return sum(distance_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

    def initialize_population(self, size, num_cities):
        # Create a list of random permutations (tours)
        return [np.random.permutation(num_cities) for _ in range(size)]

    def tournament_selection(self, population, fitnesses, k=5):
        # Select k individuals randomly and return the best (lowest fitness)
        selected = random.sample(list(zip(population, fitnesses)), k)
        return min(selected, key=lambda x: x[1])[0]

    def swap_mutation(self, individual):
        # Swap two cities in the tour to create a mutation
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
        return individual

    def order_crossover(self, parent1, parent2):
        # Perform Order Crossover (OX) to generate a child
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[a:b] = parent1[a:b]
        fill = [item for item in parent2 if item not in child]
        ptr = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = fill[ptr]
                ptr += 1
        return np.array(child)

    def eliminate_and_replace(self, population, offspring, distance_matrix):
        combined = population + offspring
        fitnesses = [self.length(ind, distance_matrix) for ind in combined]
        # Convert individuals to lists for sorting
        combined_as_lists = [ind.tolist() for ind in combined]
        sorted_combined = [np.array(ind) for _, ind in sorted(zip(fitnesses, combined_as_lists))]
        return sorted_combined[:len(population)]

    def optimize(self, filename):
        # Load the distance matrix from the CSV file
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        num_cities = distance_matrix.shape[0]
        population_size = 50
        generations = 500

        # Initialize the population
        population = self.initialize_population(population_size, num_cities)

        for generation in range(generations):
            # Evaluate fitness of each individual
            fitnesses = [self.length(ind, distance_matrix) for ind in population]
            offspring = []

            # Generate offspring through selection, crossover, and mutation
            for _ in range(population_size):
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                child = self.order_crossover(parent1, parent2)
                child = self.swap_mutation(child)
                offspring.append(child)

            # Replace the population with the best individuals
            population = self.eliminate_and_replace(population, offspring, distance_matrix)

            # Report statistics
            fitnesses = [self.length(ind, distance_matrix) for ind in population]
            mean_objective = np.mean(fitnesses)
            best_index = np.argmin(fitnesses)
            best_objective = fitnesses[best_index]
            best_solution = population[best_index]

            # Use the reporter to log progress
            time_left = self.reporter.report(mean_objective, best_objective, best_solution)
            if time_left < 0:
                break

        return 0


if __name__ == '__main__':
    # Example test run
    solver = r0926887()
    solver.optimize(
        "A:\OneDrive - KU Leuven\Master\Genetic Algorithms and Evolutionary Computing\project\GroupPhase\EvolutionaryAlgos\src\data\\tour50.csv")
