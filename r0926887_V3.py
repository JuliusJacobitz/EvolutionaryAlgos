from project.GroupPhase.EvolutionaryAlgos.src import Reporter
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt


# Student number: r0926887
class r0926887:
    def __init__(self):
        """
        Constructor for the r0926887 class.
        Initializes parameters and the reporter.
        The parameters are grouped here to make them easy to tune.
        """

        # --- EA Parameters ---

        # population_size: The number of individuals (tours) in the population.
        # - Higher value: Better exploration of the search space, less likely to
        #   get stuck in a local optimum.
        # - Lower value: Faster computation per generation.
        # - Trade-off: A larger population finds better solutions but takes more
        #   time. A good starting point is 50-100.
        self.population_size = 100

        # tournament_k: The number of individuals randomly selected for a tournament.
        # The *best* individual from this group becomes a parent.
        # - Higher value: Higher selection pressure. The best individuals are
        #   chosen more often, leading to faster convergence.
        # - Lower value: Lower selection pressure. More diversity is maintained,
        #   which can prevent premature convergence to a bad solution.
        # - Trade-off: A high 'k' (e.g., 10-20) can be good, but if it's too high,
        #   diversity is lost very quickly.
        self.tournament_k = 10

        # crossover_rate: The probability that two parents will "mate" to
        # create a new child (offspring).
        # - Higher value (e.g., 0.9): More exploration and mixing of good
        #   solution parts. This is the main driver of the EA.
        # - Lower value: More of the parents pass through to the next
        #   generation unchanged (cloning).
        self.crossover_rate = 0.9

        # mutation_rate: The probability that an individual will undergo mutation.
        # - Higher value: Introduces more random changes, increasing diversity
        #   and helping to escape local optima.
        # - Lower value: Preserves good solutions, but can lead to stagnation.
        # - Trade-off: Essential for innovation, but if too high, it turns
        #   the search into a random walk. 0.1-0.2 is often a good range.
        self.mutation_rate = 0.2

        # --- Visualization Control ---
        # Set to True to show a live convergence plot during optimization.
        self.visualize = False

        # --- Internal State (Initialized in optimize) ---
        # The reporter is no longer initialized here.
        # It will be initialized in optimize() to create unique filenames.
        self.reporter = None
        self.best_history = []
        self.mean_history = []
        self.fig = None
        self.ax = None
        self.line_best = None
        self.line_mean = None

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
        Select an individual using k-tournament selection.
        A random subset of 'k' individuals is chosen from the population.
        The one with the best (lowest) fitness in that subset is returned.
        This is a common and effective selection method.
        """
        # Get 'k' random indices from the population
        selected_indices = np.random.choice(len(population), k, replace=False)

        best_index = -1
        min_fitness = float('inf')

        # Find the best individual among the 'k' selected
        for index in selected_indices:
            if fitnesses[index] < min_fitness:
                min_fitness = fitnesses[index]
                best_index = index
        return population[best_index]

    def inversion_mutation(self, individual):
        """
        Perform inversion mutation.
        Two cut points are chosen, and the segment of the tour between them
        is reversed. This is highly effective for TSP because it's a
        "path-preserving" mutation that still creates a meaningful change.
        Example: [1, 2, |3, 4, 5,| 6] -> [1, 2, |5, 4, 3,| 6]
        """
        a, b = sorted(random.sample(range(len(individual)), 2))
        individual[a:b] = individual[a:b][::-1]
        return individual

    def order_crossover(self, parent1, parent2):
        """
        Perform Order Crossover (OX1).
        This is a standard crossover for order-based problems like TSP.
        1. A random segment from parent1 is copied to the child.
        2. The remaining slots are filled with cities from parent2 in the
           order they appear, skipping cities already in the child.
        This preserves the relative order of cities from both parents.
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
        Combine parents and offspring, then select the best to survive.
        This strategy is called "elitism." It guarantees that the best
        solution found so far is never lost from one generation to the next.
        """
        combined_population = population + offspring
        fitnesses = [self.length(ind, distance_matrix) for ind in combined_population]

        # Get the indices that would sort the fitnesses in ascending order
        sorted_indices = np.argsort(fitnesses)

        # Return the top 'population_size' individuals
        return [combined_population[i] for i in sorted_indices[:self.population_size]]

    def two_opt(self, tour, distance_matrix):
        """
        Improve a tour using the 2-opt local search heuristic.
        This makes the EA a "Memetic Algorithm" (a hybrid of EA + local search).
        It works by repeatedly swapping pairs of edges (i, i+1) and (j, j+1)
        with new edges (i, j) and (i+1, j+1) *if* it shortens the tour.
        This is done by reversing the segment between i+1 and j.

        - Impact: Massively improves solution quality.
        - Trade-off: Very computationally expensive. Each generation will
          take much longer, so fewer generations will run in the time limit.
          For TSP, this trade-off is almost always worth it.
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

                    # Propose swap: tour[i:j] (i.e., from i to j-1)
                    # This effectively swaps edge (i-1, i) and (j, (j+1)%n)
                    # with (i-1, j) and (i, (j+1)%n) ...
                    # Let's simplify: check swapping edges (i, i+1) and (j, j+1)
                    # by reversing the path from i+1 to j.

                    # We are checking the swap of edges (tour[i-1], tour[i]) and (tour[j], tour[(j+1)%n])
                    # with (tour[i-1], tour[j]) and (tour[i], tour[(j+1)%n])
                    # This is achieved by reversing tour[i:j+1]

                    # Correct 2-opt: reverse segment from i to j
                    new_tour = best_tour.copy()
                    new_tour[i:j] = best_tour[i:j][::-1]  # Reverse segment from i to j-1
                    new_length = self.length(new_tour, distance_matrix)

                    if new_length < best_length:
                        best_tour = new_tour
                        best_length = new_length
                        improved = True

        return best_tour

    def _setup_visualization(self, problem_name, num_cities):
        """Internal helper to set up the live plot."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(f"EA Convergence for {problem_name} (n={num_cities})")
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Tour Length (Fitness)')
        self.ax.grid(True)

        # Initialize empty plot lines
        self.line_best, = self.ax.plot([], [], 'r-', label='Best Fitness')
        self.line_mean, = self.ax.plot([], [], 'b--', label='Mean Fitness')

        self.ax.legend()
        plt.show()

    def _update_visualization(self):
        """Internal helper to update the live plot each generation."""
        # Update the data of the plot lines
        self.line_best.set_data(range(len(self.best_history)), self.best_history)
        self.line_mean.set_data(range(len(self.mean_history)), self.mean_history)

        # Rescale the axes
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # A small pause to allow the plot to update

    def optimize(self, filename):
        """
        Main optimization loop for the evolutionary algorithm.
        """
        # --- 1. Initialization ---
        try:
            distance_matrix = np.loadtxt(filename, delimiter=',')
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            return -1

        num_cities = distance_matrix.shape[0]

        # --- 2. Setup Reporter (with unique filename) ---
        problem_name = os.path.splitext(os.path.basename(filename))[0]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_filename = f"{self.__class__.__name__}_{problem_name}_n{num_cities}_{timestamp}"

        self.reporter = Reporter.Reporter(report_filename)

        # Write parameter header to the CSV file for traceability
        param_summary = (f"# Parameters: population_size={self.population_size}, "
                         f"tournament_k={self.tournament_k}, "
                         f"crossover_rate={self.crossover_rate}, "
                         f"mutation_rate={self.mutation_rate}\n")

        try:
            with open(self.reporter.filename, "a") as f:
                f.write(param_summary)
        except IOError as e:
            print(f"Error writing parameters to report file: {e}")

        # --- 3. Setup Visualization ---
        if self.visualize:
            self._setup_visualization(problem_name, num_cities)

        # --- 4. Initialize Population ---
        population = self.initialize_population(self.population_size, num_cities)

        # --- 5. Main Evolutionary Loop ---
        while True:
            # Evaluate fitness of each individual
            fitnesses = [self.length(ind, distance_matrix) for ind in population]
            offspring = []

            # Generate offspring
            for _ in range(self.population_size):
                parent1 = self.tournament_selection(population, fitnesses, k=self.tournament_k)
                parent2 = self.tournament_selection(population, fitnesses, k=self.tournament_k)

                # Crossover
                child = parent1.copy()
                if random.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.inversion_mutation(child)

                # Local Search (Memetic Algorithm)
                child = self.two_opt(child, distance_matrix)

                offspring.append(child)

            # Replace the old population with the best of the combined population and offspring
            population = self.eliminate_and_replace(population, offspring, distance_matrix)

            # --- 6. Report Statistics ---
            current_fitnesses = [self.length(ind, distance_matrix) for ind in population]
            mean_objective = np.mean(current_fitnesses)
            best_index = np.argmin(current_fitnesses)
            best_objective = current_fitnesses[best_index]
            best_solution = population[best_index]

            # --- 7. Update Visualization ---
            if self.visualize:
                self.best_history.append(best_objective)
                self.mean_history.append(mean_objective)
                self._update_visualization()

            # --- 8. Report to File and Check Time ---
            time_left = self.reporter.report(mean_objective, best_objective, best_solution)
            if time_left < 0:
                print("Time limit reached. Stopping optimization.")
                break

        # --- 9. Finalization ---
        print(f"Optimization Finished! Results saved to {self.reporter.filename}")
        if self.visualize:
            plt.ioff()
            plot_filename = self.reporter.filename.replace('.csv', '.png')
            try:
                self.fig.savefig(plot_filename)
                print(f"Convergence plot saved to {plot_filename}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            plt.show()  # Keep the final plot window open
        return 0


if __name__ == '__main__':
    # Create an instance of the solver
    solver = r0926887()

    # Run the optimization on the tour50 problem
    # *** IMPORTANT: Make sure 'tour50.csv' is in the same directory
    # or provide the full, correct path. ***
    problem_file = "./src/data/tour250.csv"

    if not os.path.exists(problem_file):
        print(f"Error: File '{problem_file}' not found.")
        print("Please make sure it is in the same folder as the script.")
    else:
        solver.optimize(problem_file)