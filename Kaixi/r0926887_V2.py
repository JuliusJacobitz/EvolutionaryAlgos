import Reporter
import numpy as np
import random
import time


# Student number: r0926887
class r0926887:
    def __init__(self):
        """
        Constructor for the r0926887 class.
        Initializes parameters and the reporter.
        The parameters are grouped here to make them easy to tune.
        """
        # --- EA Parameters ---
        self.population_size = 100  # Number of individuals in the population
        self.tournament_k = 10  # Size of the tournament for selection
        self.crossover_rate = 0.9  # Probability of performing crossover
        self.mutation_rate = 0.2  # Probability of mutating an individual

        # --- Reporting ---
        # The reporter for logging progress. The filename is now auto-generated
        # to reflect the parameters used in this run.
        self.reporter = Reporter.Reporter(self.__class__.__name__)

        # Add a header to the CSV file with the parameters for this run.
        # This is done by appending to the file right after the Reporter creates it.
        # This helps in tracking which parameters led to which results.
        param_summary = (f"# Parameters: population_size={self.population_size}, "
                         f"tournament_k={self.tournament_k}, "
                         f"crossover_rate={self.crossover_rate}, "
                         f"mutation_rate={self.mutation_rate}\n")
        try:
            with open(self.reporter.filename, "a") as f:
                f.write(param_summary)
        except IOError as e:
            print(f"Error writing parameters to report file: {e}")

    def length(self, tour, distance_matrix):
        """
        Calculate the total length of a TSP tour (cycle).
        """
        # The tour is a permutation of city indices. The total length is the sum
        # of distances between consecutive cities in the tour, including the
        # distance from the last city back to the first.
        return sum(distance_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

    def initialize_population(self, size, num_cities):
        """
        Create an initial population of random tours.
        """
        return [np.random.permutation(num_cities) for _ in range(size)]

    def tournament_selection(self, population, fitnesses, k):
        """
        Select an individual using tournament selection.
        A random subset of k individuals is chosen, and the best one is returned.
        """
        selected_indices = np.random.choice(len(population), k, replace=False)
        best_index = -1
        min_fitness = float('inf')
        for index in selected_indices:
            if fitnesses[index] < min_fitness:
                min_fitness = fitnesses[index]
                best_index = index
        return population[best_index]

    def inversion_mutation(self, individual):
        """
        Perform inversion mutation.
        This is generally more effective for TSP than swap mutation because it
        reverses a sub-sequence, which is a larger, more meaningful change
        that still preserves much of the tour's structure.
        """
        a, b = sorted(random.sample(range(len(individual)), 2))
        individual[a:b] = individual[a:b][::-1]
        return individual

    def order_crossover(self, parent1, parent2):
        """
        Perform Order Crossover (OX1).
        """
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))

        # Initialize child with a placeholder
        child = [-1] * size

        # Copy the segment from parent1 to the child
        child[a:b] = parent1[a:b]

        # Fill the rest of the child with genes from parent2
        parent2_pos = b
        child_pos = b
        while -1 in child:
            if parent2[parent2_pos] not in child:
                child[child_pos] = parent2[parent2_pos]
                child_pos = (child_pos + 1) % size
            parent2_pos = (parent2_pos + 1) % size

        return np.array(child)

    def eliminate_and_replace(self, population, offspring, distance_matrix):
        """
        Combine parents and offspring, then select the best individuals to survive.
        This is a form of elitism, ensuring the best solutions are not lost.
        """
        combined_population = population + offspring
        fitnesses = [self.length(ind, distance_matrix) for ind in combined_population]
        sorted_indices = np.argsort(fitnesses)

        # Return the top 'population_size' individuals
        return [combined_population[i] for i in sorted_indices[:self.population_size]]

    def two_opt(self, tour, distance_matrix):
        """
        Improve a tour using the 2-opt local search heuristic.
        It repeatedly swaps pairs of edges to find a better tour until no
        more improvements can be made. This turns the GA into a Memetic Algorithm.
        """
        n = len(tour)
        improved = True
        best_tour = tour.copy()
        best_length = self.length(best_tour, distance_matrix)

        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1: continue  # Skip adjacent edges

                    # Propose a 2-opt swap: reverse the segment between i and j
                    new_tour = best_tour.copy()
                    new_tour[i:j] = best_tour[i:j][::-1]
                    new_length = self.length(new_tour, distance_matrix)

                    if new_length < best_length:
                        best_tour = new_tour
                        best_length = new_length
                        improved = True

        return best_tour

    def optimize(self, filename):
        """
        Main optimization loop for the evolutionary algorithm.
        """
        # Read distance matrix from file.
        distance_matrix = np.loadtxt(filename, delimiter=',')
        num_cities = distance_matrix.shape[0]

        # Initialize population
        population = self.initialize_population(self.population_size, num_cities)

        # Main loop
        while True:
            # Evaluate fitness of each individual
            fitnesses = [self.length(ind, distance_matrix) for ind in population]
            offspring = []

            # Generate offspring
            for _ in range(self.population_size):
                parent1 = self.tournament_selection(population, fitnesses, k=self.tournament_k)
                parent2 = self.tournament_selection(population, fitnesses, k=self.tournament_k)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.inversion_mutation(child)

                # Local Search (2-Opt)
                # Applying a local search heuristic makes this a Memetic Algorithm,
                # which can significantly improve performance for TSP.
                child = self.two_opt(child, distance_matrix)

                offspring.append(child)

            # Replace the old population with the best of the combined population and offspring
            population = self.eliminate_and_replace(population, offspring, distance_matrix)

            # Report statistics for the current generation
            current_fitnesses = [self.length(ind, distance_matrix) for ind in population]
            mean_objective = np.mean(current_fitnesses)
            best_index = np.argmin(current_fitnesses)
            best_objective = current_fitnesses[best_index]
            best_solution = population[best_index]

            # Use the reporter to log progress and check time limit
            time_left = self.reporter.report(mean_objective, best_objective, best_solution)
            if time_left < 0:
                break

        return 0


if __name__ == '__main__':
    # Create an instance of the solver
    solver = r0926887()
    # Run the optimization on the tour50 problem
    # Note: Update the path to your tour50.csv file if necessary
    solver.optimize(
        "A:\OneDrive - KU Leuven\Master\Genetic Algorithms and Evolutionary Computing\project\GroupPhase\EvolutionaryAlgos\src\data\\tour50.csv")
