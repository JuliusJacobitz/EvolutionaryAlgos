# use fast_tsp to solve it semi-optimally to get a benchmark

import pandas as pd
import fast_tsp

# Load and clean the CSV
tournumber = 50
file_path = f"/Users/julius/Library/CloudStorage/GoogleDrive-juliusjacobitz@gmail.com/My Drive/Studium/Master/07_Semester_Leuven/Genetic Algorithms/CodeGroupPhase/src/data/tour{tournumber}.csv"
dist_matrix = pd.read_csv(file_path, header=None)

# Ensure it's a proper square matrix
dist_matrix = dist_matrix.apply(pd.to_numeric, errors='coerce')
dist_matrix = dist_matrix.dropna(axis=1, how='all')
dist_matrix = dist_matrix.dropna(axis=0, how='all')
n = min(dist_matrix.shape)
dist_matrix = dist_matrix.iloc[:n, :n].values

# Round distances to integers
dist_matrix = dist_matrix.round().astype(int)

# Solve the TSP
tour = fast_tsp.find_tour(dist_matrix)

print("Optimal (approximate) tour order:")
print(tour)

# Compute total distance
total_distance = sum(
    dist_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))
)
print(f"\nTotal tour length: {total_distance}")
