# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:53:33 2024

@author: Jaime
"""

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Definición del problema
NUM_NODOS = 50
CAPACIDAD_MAXIMA = 100
DEMANDA = np.random.randint(1, 10, size=NUM_NODOS)
COSTO_ENERGIA = np.random.rand(NUM_NODOS, NUM_NODOS)
PÉRDIDAS_ENERGIA = np.random.rand(NUM_NODOS, NUM_NODOS) * 0.1

# Crear la clase Fitness y el Individuo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def crear_individuo():
    return creator.Individual(np.random.permutation(NUM_NODOS))

def evaluar_individuo(individuo):
    costo_total = 0
    perdidas_totales = 0

    for i in range(NUM_NODOS - 1):
        nodo_actual = individuo[i]
        nodo_siguiente = individuo[i + 1]

        costo_total += COSTO_ENERGIA[nodo_actual][nodo_siguiente]
        perdidas_totales += PÉRDIDAS_ENERGIA[nodo_actual][nodo_siguiente] * DEMANDA[nodo_siguiente]

    return costo_total + perdidas_totales,

toolbox = base.Toolbox()
toolbox.register("attr_ind", crear_individuo)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluar_individuo)

def main():
    random.seed(42)
    pop = toolbox.population(n=100)
    NGEN = 50
    CXPB, MUTPB = 0.5, 0.2
    
    # Algoritmo evolutivo
    for gen in range(NGEN):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = map(toolbox.evaluate, offspring)
        
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        pop = toolbox.select(offspring, k=len(pop))
        record = tools.selBest(pop, 1)[0]
        print(f"Generación {gen}: Mejor solución -> {record.fitness.values[0]}")
    
    mejor_solucion = tools.selBest(pop, 1)[0]
    print("Mejor configuración encontrada: ", mejor_solucion)
    print("Costo total con pérdidas: ", mejor_solucion.fitness.values[0])

    return mejor_solucion

# Ejecutar el algoritmo genético y obtener la mejor solución
mejor_solucion = main()

# Crear el grafo basado en la mejor solución
G = nx.DiGraph()

# Añadir los nodos y aristas según la mejor solución
for i in range(NUM_NODOS - 1):
    nodo_actual = mejor_solucion[i]
    nodo_siguiente = mejor_solucion[i + 1]
    G.add_edge(nodo_actual, nodo_siguiente, weight=COSTO_ENERGIA[nodo_actual][nodo_siguiente])

# Dibujar el grafo
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
plt.title("Mejor Configuración de la Red de Distribución de Energía")
plt.show()
