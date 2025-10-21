
import pandas as pd
import plotly.graph_objects as go

# Load the convergence data from the CSV file
filename = "r0123456.csv"
df = pd.read_csv(filename, comment='#')

# Extract relevant columns
iterations = df.iloc[:, 0]
elapsed_time = df.iloc[:, 1]
mean_objective = df.iloc[:, 2]
best_objective = df.iloc[:, 3]

# Create the convergence plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=iterations, y=mean_objective, mode='lines+markers', name='Mean Objective'))
fig.add_trace(go.Scatter(x=iterations, y=best_objective, mode='lines+markers', name='Best Objective'))

fig.update_layout(
    title="Convergence Graph",
    xaxis_title="Iteration",
    yaxis_title="Objective Value",
    legend_title="Metrics",
    template="plotly_white"
)

# Save the plot
fig.write_image("convergence_graph.png")
fig.write_json("convergence_graph.json")

print("Convergence graph saved as 'convergence_graph.png' and 'convergence_graph.json'.")
