# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:53:58 2023

@author: Jaime
"""
#==============================================================================
# APLICACIÓN DE UN ALGORITMO GENETICO EN UNA RED DE DISTRIBUCIÓN ELÉCTRICA

# Este código aplica un algoritmo genético para optimizar la configuración de 
# interruptores en una red de distribución eléctrica modelada como un grafo 
# (network). El objetivo es encontrar la mejor configuración de interruptores 
# cerrados (representados por bits en un vector binario) que maximice una 
# función de aptitud, que en este caso es simplemente el número total de 
# interruptores cerrados. Se crea una población inicial de posibles 
# configuraciones, y a lo largo de 100 generaciones, se seleccionan los 
# mejores individuos de la población (es decir, las configuraciones con más 
# interruptores cerrados), se realizan cruces entre ellos para generar 
# descendencia, y se aplican mutaciones aleatorias para explorar nuevas 
# soluciones. El mejor individuo encontrado al final del proceso representa 
# la configuración óptima de interruptores cerrados, que luego se visualiza en 
# un gráfico de la red. Además, en la versión extendida del código con 100 
# nodos, se crea una red más grande, se aplica el mismo proceso genético, y 
# se presenta la configuración óptima en un gráfico más detallado, junto con 
# una tabla que muestra el estado (abierto o cerrado) de cada interruptor en 
# la red.
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define the network
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 15), (4,6), (5,6), (6, 8), (9,8), (9, 10)])

# Define the fitness function
def fitness(individual):
    # In this simple example, the fitness is just the sum of the bits (i.e., the number of closed switches)
    # In a real-world application, you would need to run a power flow simulation and calculate the actual power losses
    return np.sum(individual)

# Initialize the population
population_size = 30
population = np.random.randint(2, size=(population_size, 11))

# Run the genetic algorithm
num_generations = 100
for _ in range(num_generations):
    # Calculate fitness values
    fitness_values = np.array([fitness(individual) for individual in population])

    # Select the best individuals
    best_individuals = population[np.argsort(fitness_values)[-population_size//2:]]

    # Crossover
    for _ in range(population_size // 2):
        parent1, parent2 = np.random.choice(len(best_individuals), 2, replace=False)
        crossover_point = np.random.randint(11)
        offspring = np.concatenate((best_individuals[parent1][:crossover_point], best_individuals[parent2][crossover_point:]))
        population = np.vstack((population, offspring))

    # Mutation
    for i in range(len(population)):
        if np.random.rand() < 0.1:
            mutation_point = np.random.randint(11)
            population[i, mutation_point] ^= 1  # Flip the bit at the mutation point

# Print the best solution
best_solution = population[np.argmax(fitness_values)]
print('Best solution:', best_solution)

# Generate the layout
layout = nx.spring_layout(G)

# Draw the network
fig, ax = plt.subplots()
nx.draw(G, pos=layout, with_labels=True, node_color='lightblue', font_weight='bold', ax=ax)

closed_switches = [i for i, x in enumerate(best_solution) if x == 1]
highlighted_edges = [list(G.edges)[i] for i in closed_switches]
nx.draw_networkx_edges(G, pos=layout, edgelist=highlighted_edges, width=3, edge_color='blue', ax=ax)
plt.show()

#==============================================================================
# APLICACIÓN DE UN ALGORITMO GENETICO EN UNA RED DE DISTRIBUCIÓN ELÉCTRICA CON 
# 100 NODOS
#==============================================================================
#####
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

# Define the fitness function
def fitness(individual):
    return np.sum(individual)

# Initialize the population
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
fig, ax = plt.subplots(figsize=(15, 15))  # Increase the figure size
nx.draw(G, pos=shell_layout, with_labels=True, node_color='lightblue', font_size=8, font_weight='bold', ax=ax)  # Reduce font size

closed_switches = [i for i, x in enumerate(best_solution) if x == 1]
highlighted_edges = [list(G.edges)[i] for i in closed_switches]
nx.draw_networkx_edges(G, pos=shell_layout, edgelist=highlighted_edges, width=3, edge_color='blue', ax=ax)

plt.show()

import pandas as pd

# Prepare data for the table
edge_data = []
for i, edge_status in enumerate(best_solution):
    from_node, to_node = list(G.edges)[i]
    status = "Closed" if edge_status == 1 else "Open"
    edge_data.append((from_node, to_node, status))

# Create a DataFrame
edge_df = pd.DataFrame(edge_data, columns=["From Node", "To Node", "Status"])

# Print the table
print(edge_df)








