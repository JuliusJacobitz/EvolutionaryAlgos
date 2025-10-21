
import Reporter
import numpy as np
import random

class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def evaluate(self, tour, distanceMatrix):
        return sum(distanceMatrix[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

    def initialize_population(self, pop_size, num_cities):
        return [np.random.permutation(num_cities) for _ in range(pop_size)]

    def selection(self, population, fitnesses):
        selected = random.choices(population, weights=[1/f for f in fitnesses], k=len(population))
        return selected

    def mutate(self, tour, mutation_count=3):
        for _ in range(mutation_count):
            i, j = sorted(random.sample(range(len(tour)), 2))
            tour[i:j+1] = list(reversed(tour[i:j+1]))
        return tour

    def recombine(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1]*size
        child[start:end] = parent1[start:end]
        fill = [city for city in parent2 if city not in child]
        idx = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return np.array(child)

    def eliminate(self, old_pop, new_pop, old_fit, new_fit, elite_count):
        combined = list(zip(old_pop + new_pop, old_fit + new_fit))
        combined.sort(key=lambda x: x[1])
        survivors = [ind for ind, fit in combined[:len(old_pop)]]
        return survivors[:elite_count] + survivors[elite_count:]

    def optimize(self, filename):
        distanceMatrix = np.loadtxt(filename, delimiter=",")
        num_cities = len(distanceMatrix)
        pop_size = 100
        MAX_ITER = 2000
        NO_IMPROVE_LIMIT = 200
        ELITE_COUNT = 1
        MUTATION_COUNT = 1

        population = self.initialize_population(pop_size, num_cities)
        fitnesses = [self.evaluate(ind, distanceMatrix) for ind in population]

        bestObjective = min(fitnesses)
        bestSolution = population[np.argmin(fitnesses)]
        no_improve_counter = 0

        for iteration in range(MAX_ITER):
            meanObjective = np.mean(fitnesses)
            current_best_idx = np.argmin(fitnesses)
            current_bestObjective = fitnesses[current_best_idx]
            current_bestSolution = population[current_best_idx]

            timeLeft = self.reporter.report(meanObjective, current_bestObjective, current_bestSolution)
            if timeLeft < 0:
                break

            if current_bestObjective < bestObjective:
                bestObjective = current_bestObjective
                bestSolution = current_bestSolution
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter >= NO_IMPROVE_LIMIT:
                break

            selected = self.selection(population, fitnesses)
            offspring = []
            for i in range(0, len(selected), 2):
                if i+1 < len(selected):
                    child = self.recombine(selected[i], selected[i+1])
                    child = self.mutate(child, mutation_count=MUTATION_COUNT)
                    offspring.append(child)

            new_fitnesses = [self.evaluate(ind, distanceMatrix) for ind in offspring]
            population = self.eliminate(population, offspring, fitnesses, new_fitnesses, ELITE_COUNT)
            fitnesses = [self.evaluate(ind, distanceMatrix) for ind in population]

        return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python solution_template.py <path_to_csv_file>")
    else:
        solver = r0123456()
        solver.optimize(sys.argv[1])



