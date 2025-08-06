# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:02:14 2024

@author: Jaime
"""
#==============================================================================
# PROBLEMA DE LA MOCHILA RESUELTO CON 10 ALGORITMOS Y 25 ARTÍCULOS
#==============================================================================

# Cargar librerías
import time
import numpy as np
import random
from deap import base, creator, tools, algorithms
import pandas as pd
import matplotlib.pyplot as plt

# Paso 1: Generar datos simulados con 25 artículos
np.random.seed(42)
num_items = 25
weights = np.random.randint(1, 100, size=num_items)
values = np.random.randint(10, 500, size=num_items)
capacity = 500

# Función para medir el tiempo de ejecución del algoritmo genético
def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start

# Función para el algoritmo genético
def genetic_algorithm_knapsack(weights, values, capacity, population_size, cxpb, mutpb, generations):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(weights))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_knapsack(individual):
        total_weight = sum(individual[i] * weights[i] for i in range(len(individual)))
        if total_weight > capacity:
            return 0,
        total_value = sum(individual[i] * values[i] for i in range(len(individual)))
        return total_value,
    
    toolbox.register("evaluate", eval_knapsack)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, verbose=False)
    
    best_individual = tools.selBest(population, 1)[0]
    best_value = eval_knapsack(best_individual)[0]
    return best_value

# Parámetros a probar
population_sizes = [400, 800]
crossover_probs = [0.5]
mutation_probs = [0.1]
generations = [50, 100, 200]

# Almacenar resultados
results = []

# Ejecutar el algoritmo genético con diferentes configuraciones de parámetros
for pop_size in population_sizes:
    for cxpb in crossover_probs:
        for mutpb in mutation_probs:
            for gen in generations:
                best_value, exec_time = measure_time(
                    genetic_algorithm_knapsack, weights, values, capacity, pop_size, cxpb, mutpb, gen
                )
                results.append({
                    'Población': pop_size,
                    'Probabilidad de Cruce': cxpb,
                    'Probabilidad de Mutación': mutpb,
                    'Generaciones': gen,
                    'Valor Máximo': best_value,
                    'Tiempo de Ejecución (s)': exec_time
                })

# Convertir los resultados a un DataFrame
df_results = pd.DataFrame(results)

# Mostrar los resultados
print(df_results)

# Graficar los resultados
plt.figure(figsize=(12, 6))
for pop_size in population_sizes:
    subset = df_results[df_results['Población'] == pop_size]
    plt.plot(subset['Generaciones'], subset['Valor Máximo'], marker='o', label=f'Población {pop_size}')

plt.xlabel('Número de generaciones')
plt.ylabel('Valor máximo obtenido')
plt.title('Algoritmo genético: Valor máximo vs. generaciones')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for pop_size in population_sizes:
    subset = df_results[df_results['Población'] == pop_size]
    plt.plot(subset['Generaciones'], subset['Tiempo de Ejecución (s)'], marker='o', label=f'Población {pop_size}')

plt.xlabel('Número de generaciones')
plt.ylabel('Tiempo de ejecución (s)')
plt.title('Algoritmo genético: Tiempo de Ejecución vs. Generaciones')
plt.legend()
plt.grid(True)
plt.show()
