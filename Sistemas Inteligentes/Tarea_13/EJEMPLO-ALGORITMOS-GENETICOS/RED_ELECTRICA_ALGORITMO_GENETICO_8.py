# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:26:23 2023

@author: Jaime
"""
#==============================================================================
# APLICACIÓN DE UN ALGORITMO GENETICO EN UNA RED DE DISTRIBUCIÓN ELÉCTRICA
# Una mejora esencial sería implementar una simulación
# de flujo de potencia que tome en cuenta las restricciones de la red, como
# los límites de capacidad, voltaje, y balance de carga y generación.

# La función de fitness ahora incluye una verificación de las restricciones 
# de capacidad, y puedes expandirla fácilmente para incluir otros límites 
# como los de voltaje y balance de carga y generación.
#==============================================================================
# !pip install pandapower

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp
import pandas as pd

# Define the network using Pandapower
net = pp.create_empty_network()
buses = [pp.create_bus(net, vn_kv=20.) for _ in range(100)]
pp.create_ext_grid(net, buses[0])

# Define some random connections between the buses
np.random.seed(42)  # For reproducibility
for i in range(100):  # Modified to 100
    from_bus = i
    to_bus = np.random.randint(100)  # Modified to 100
    if from_bus != to_bus:
        pp.create_line(net, buses[from_bus], buses[to_bus], length_km=np.random.rand() * 2, std_type="NAYY 4x50 SE")

# Define the fitness function
def fitness(individual):
    for i, line_status in enumerate(individual):
        net.line.at[i, 'in_service'] = bool(line_status)

    pp.runpp(net)

    fitness_value = 0
    loading_percent = net.res_line.loading_percent
    for loading in loading_percent:
        if loading < 100: # Assuming 100% as the capacity limit
            fitness_value += 1
        else:
            fitness_value -= 1

    return fitness_value

# Initialize the population
population_size = 30
population = np.random.randint(2, size=(population_size, len(net.line)))

# Run the genetic algorithm
num_generations = 100
for _ in range(num_generations):
    fitness_values = np.array([fitness(individual) for individual in population])
    best_individuals = population[np.argsort(fitness_values)[-population_size//2:]]
    population = best_individuals.copy()

    # Crossover
    for _ in range(population_size // 2):
        parent1, parent2 = np.random.choice(len(best_individuals), 2, replace=False)
        crossover_point = np.random.randint(len(net.line))
        offspring = np.concatenate((best_individuals[parent1][:crossover_point], best_individuals[parent2][crossover_point:]))
        population = np.vstack((population, offspring))

    # Mutation
    for i in range(len(population)):
        if np.random.rand() < 0.1:
            mutation_point = np.random.randint(len(net.line))
            population[i, mutation_point] ^= 1

# Print the best solution
best_solution = population[np.argmax(fitness_values)]
print('Best solution:', best_solution)

# Generate a results table
results_df = pd.DataFrame({
    'Line ID': range(len(net.line)),
    'From Bus': net.line['from_bus'],
    'To Bus': net.line['to_bus'],
    'Status': best_solution,
    'Loading Percent': net.res_line['loading_percent']
})
print("Results Table:")
print(results_df)

# Generate NetworkX Graph for visualization
G = nx.Graph()
G.add_edges_from(net.line[["from_bus", "to_bus"]].values.tolist())

# Generate the layout using a shell layout for better separation
shell_layout = nx.shell_layout(G)

# Draw the network
fig, ax = plt.subplots(figsize=(15, 15))  # Increase the figure size
nx.draw(G, pos=shell_layout, with_labels=True, node_color='lightblue', font_size=8, font_weight='bold', ax=ax)  # Reduce font size

closed_switches = [i for i, x in enumerate(best_solution) if x == 1]
highlighted_edges = [list(G.edges)[i] for i in closed_switches]
nx.draw_networkx_edges(G, pos=shell_layout, edgelist=highlighted_edges, width=3, edge_color='blue', ax=ax)

plt.show()
