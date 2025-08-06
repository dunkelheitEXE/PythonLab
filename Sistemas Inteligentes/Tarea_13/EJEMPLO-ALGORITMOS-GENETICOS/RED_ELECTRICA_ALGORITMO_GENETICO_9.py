# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 18:50:55 2023

@author: Jaime
"""
#==============================================================================
# APLICACIÓN DE UN ALGORITMO GENETICO EN UNA RED DE DISTRIBUCIÓN ELÉCTRICA
#####
#==============================================================================
# !pip install pandapower
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Define the network
G = nx.Graph()

# Add 100 nodes
for i in range(100):
    G.add_node(i)

# Add edges randomly between nodes with associated properties
np.random.seed(42)  # For reproducibility
for i in range(100):
    from_node = i
    to_node = np.random.randint(100)
    if from_node != to_node:
        # Add transmission capacity, resistance, and reactance
        G.add_edge(from_node, to_node, capacity=np.random.uniform(10, 100), resistance=np.random.uniform(0.1, 1), reactance=np.random.uniform(0.1, 1))

# Define the fitness function
def fitness(individual):
    total_capacity = 0
    for i, edge_status in enumerate(individual):
        if edge_status == 1:
            u, v = list(G.edges)[i]
            total_capacity += G[u][v]['capacity']
    return total_capacity

# Initialize the population with edge status and associated properties
population_size = 30
population = np.random.randint(2, size=(population_size, len(G.edges)))

# Run the genetic algorithm
num_generations = 100
for _ in range(num_generations):
    fitness_values = np.array([fitness(individual) for individual in population])
    best_individuals = population[np.argsort(fitness_values)[-population_size//2:]]

    # Crossover
    for _ in range(population_size // 2):
        parent1, parent2 = np.random.choice(len(best_individuals), 2, replace=False)
        crossover_point = np.random.randint(len(G.edges))
        offspring = np.concatenate((best_individuals[parent1][:crossover_point], best_individuals[parent2][crossover_point:]))
        population = np.vstack((population, offspring))

    # Mutation
    for i in range(len(population)):
        if np.random.rand() < 0.1:
            mutation_point = np.random.randint(len(G.edges))
            population[i, mutation_point] ^= 1

# Print the best solution
best_solution = population[np.argmax(fitness_values)]
print('Best solution:', best_solution)

# Generate the layout using a shell layout for better separation
shell_layout = nx.shell_layout(G)

# Draw the network
fig, ax = plt.subplots(figsize=(15, 15))
nx.draw(G, pos=shell_layout, with_labels=True, node_color='lightblue', font_size=8, font_weight='bold', ax=ax)

closed_switches = [i for i, x in enumerate(best_solution) if x == 1]
highlighted_edges = [list(G.edges)[i] for i in closed_switches]
nx.draw_networkx_edges(G, pos=shell_layout, edgelist=highlighted_edges, width=3, edge_color='blue', ax=ax)

plt.show()

# Prepare data for the table
edge_data = []
for i, edge_status in enumerate(best_solution):
    from_node, to_node = list(G.edges)[i]
    status = "Closed" if edge_status == 1 else "Open"
    capacity, resistance, reactance = G[from_node][to_node]['capacity'], G[from_node][to_node]['resistance'], G[from_node][to_node]['reactance']
    edge_data.append((from_node, to_node, status, capacity, resistance, reactance))

# Create a DataFrame
edge_df = pd.DataFrame(edge_data, columns=["From Node", "To Node", "Status", "Capacity", "Resistance", "Reactance"])

# Print the table
print(edge_df)
