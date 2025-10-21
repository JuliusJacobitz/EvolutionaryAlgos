# info: this requires a fix in Reporter, which changes the delimiter from "," to "-" when writng the cycle

import pandas as pd
import matplotlib.pyplot as plt
folder = "/Users/julius/Library/CloudStorage/GoogleDrive-juliusjacobitz@gmail.com/My Drive/Studium/Master/07_Semester_Leuven/Genetic Algorithms/CodeGroupPhase/src/data/output_julius/"
tour= "250"
filepath = f"{tour}/tour_{tour}_julius_1761048067.csv"
df = pd.read_csv(folder+filepath,skiprows=1)
df.columns = [c.strip() for c in df.columns]


# Plot the data
# x axis = "# Iteration", y axis = "Mean value"
plt.figure(figsize=(10, 6))
plt.plot(df["# Iteration"], df["Mean value"], label="Mean Value", color='blue')
plt.plot(df["# Iteration"], df["Best value"], label="Best Value", color='orange')
plt.ylim(80000, 150000)
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Mean and Best Value over Iterations")
plt.legend()
plt.grid()
plt.show()

