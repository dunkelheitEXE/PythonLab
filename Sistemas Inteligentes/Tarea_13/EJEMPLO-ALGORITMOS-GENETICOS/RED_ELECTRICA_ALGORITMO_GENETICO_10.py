# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:58:47 2023

@author: Jaime
"""

#==============================================================================
# APLICACIÓN DE UN ALGORITMO GENETICO PARA LA MINIMIZACIÓN DE PÉRDIDAS DE ENERGÍA
# EN UNA RED DE DISTRIBUCIÓN ELÉCTRICA CON 100 NODOS
# Este código utiliza un algoritmo genético para encontrar una configuración de
# interruptores que minimice las pérdidas de energía en una red de distribución
# eléctrica. La función de aptitud se ha diseñado para representar las pérdidas
# de energía como la suma de los interruptores abiertos, y el algoritmo busca
# minimizar esta suma. Los interruptores abiertos se destacan en rojo en 
# la visualización de la red.
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define the network
G = nx.Graph()

# Add 100 nodes
for i in range(100):
    G.add_node(i)

# Add edges randomly between nodes
np.random.seed(42)  # For reproducibility
for i in range(100):
    from_node = i
    to_node = np.random.randint(100)
    if from_node != to_node:
        G.add_edge(from_node, to_node)

# Define the fitness function for minimizing energy loss
def fitness(individual):
    # Simulate energy loss as the sum of open switches
    # In a real-world application, you would need to run a power flow simulation
    return -np.sum(individual)

# Initialize the population
population_size = 30
population = np.random.randint(2, size=(population_size, len(G.edges)))

# Run the genetic algorithm
num_generations = 100
for _ in range(num_generations):
    fitness_values = np.array([fitness(individual) for individual in population])
    best_individuals = population[np.argsort(fitness_values)[:population_size//2]]  # Select the worst individuals

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
best_solution = population[np.argmin(fitness_values)]  # Get the minimum fitness value
print('Best solution:', best_solution)

# Generate the layout using a shell layout for better separation
shell_layout = nx.shell_layout(G)

# Draw the network
fig, ax = plt.subplots(figsize=(15, 15))  # Increase the figure size
nx.draw(G, pos=shell_layout, with_labels=True, node_color='lightblue', font_size=8, font_weight='bold', ax=ax)  # Reduce font size

open_switches = [i for i, x in enumerate(best_solution) if x == 0]
highlighted_edges = [list(G.edges)[i] for i in open_switches]
nx.draw_networkx_edges(G, pos=shell_layout, edgelist=highlighted_edges, width=3, edge_color='red', ax=ax)

plt.show()

import pandas as pd

# Prepare data for the table
edge_data = []
for i, edge_status in enumerate(best_solution):
    from_node, to_node = list(G.edges)[i]
    status = "Open" if edge_status == 0 else "Closed"
    edge_data.append((from_node, to_node, status))

# Create a DataFrame
edge_df = pd.DataFrame(edge_data, columns=["From Node", "To Node", "Status"])

# Print the table
print(edge_df)
