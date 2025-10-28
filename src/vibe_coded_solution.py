import Reporter
import numpy as np
import random
import time
import os

class r0123456:

    def __init__(self, output_file):
        self.reporter = Reporter.Reporter(filename=output_file)
        self.populationSize = 300
        self.kTournament = 5
        self.elitismRate = 0.05
        self.maxGenerations = 2000
        self.mutationRate = 0.2

    def evaluate(self, solution, distanceMatrix):
        return sum(distanceMatrix[solution[i], solution[(i+1)%len(solution)]] for i in range(len(solution)))

    def initialize_population(self, numCities, distanceMatrix):
        population = []
        for _ in range(self.populationSize):
            solution = list(np.random.permutation(numCities))
            population.append(solution)
        return population

    def k_tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(self.populationSize):
            contestants = random.sample(range(len(population)), self.kTournament)
            best = min(contestants, key=lambda i: fitnesses[i])
            selected.append(population[best])
        return selected

    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1]*size
        child[start:end] = parent1[start:end]
        p2_index = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[p2_index] in child:
                    p2_index += 1
                child[i] = parent2[p2_index]
        return child

    def mutate(self, solution):
        if random.random() < self.mutationRate:
            i, j = sorted(random.sample(range(len(solution)), 2))
            solution[i:j] = reversed(solution[i:j])
        return solution

    def eliminate(self, population, fitnesses, offspring, offspringFitnesses):
        numElite = int(self.elitismRate * self.populationSize)
        eliteIndices = np.argsort(fitnesses)[:numElite]
        elite = [population[i] for i in eliteIndices]
        eliteFitnesses = [fitnesses[i] for i in eliteIndices]

        combined = offspring + elite
        combinedFitnesses = offspringFitnesses + eliteFitnesses

        newPopulation = []
        for _ in range(self.populationSize):
            contestants = random.sample(range(len(combined)), self.kTournament)
            best = min(contestants, key=lambda i: combinedFitnesses[i])
            newPopulation.append(combined[best])
        return newPopulation

    def optimize(self, filename):
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        numCities = distanceMatrix.shape[0]
        population = self.initialize_population(numCities, distanceMatrix)
        fitnesses = [self.evaluate(ind, distanceMatrix) for ind in population]

        bestObjective = min(fitnesses)
        bestSolution = population[np.argmin(fitnesses)]
        noImprovementCounter = 0

        for generation in range(self.maxGenerations):
            selected = self.k_tournament_selection(population, fitnesses)
            offspring = []

            for i in range(0, self.populationSize, 2):
                p1, p2 = selected[i], selected[(i + 1) % self.populationSize]
                child1 = self.order_crossover(p1, p2)
                child2 = self.order_crossover(p2, p1)
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))

            offspringFitnesses = [self.evaluate(ind, distanceMatrix) for ind in offspring]
            population = self.eliminate(population, fitnesses, offspring, offspringFitnesses)
            fitnesses = [self.evaluate(ind, distanceMatrix) for ind in population]

            currentBest = min(fitnesses)
            currentBestSolution = population[np.argmin(fitnesses)]
            meanObjective = np.mean(fitnesses)

            if currentBest < bestObjective:
                bestObjective = currentBest
                bestSolution = currentBestSolution
                noImprovementCounter = 0
            else:
                noImprovementCounter += 1

            timeLeft = self.reporter.report(meanObjective, bestObjective, np.array(bestSolution))
            if timeLeft < 0 or noImprovementCounter > 50:
                break

        return bestSolution


csv_path = "data/tour50.csv"
solver = r0123456(output_file="data/output/50" + "/tour_50_"+str(int(time.time())).split(".")[0])
result = solver.optimize(csv_path)

# Convert solution to plain integers
best_solution = [int(city) for city in result]

# Load distance matrix again to evaluate the objective
distance_matrix = np.loadtxt(csv_path, delimiter=",")
best_objective = solver.evaluate(best_solution, distance_matrix)

print("Best tour:", best_solution)
print("Best objective value:", best_objective)

