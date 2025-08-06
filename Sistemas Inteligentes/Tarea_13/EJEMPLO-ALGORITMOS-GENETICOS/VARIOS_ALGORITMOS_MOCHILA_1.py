# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:36:17 2024

@author: Jaime
"""
#==============================================================================
# PROBLEMA DE LA MOCHILA RESUELTO CON 10 ALGORITMOS Y 5 ARTÍCULOS

# Explicación del código
# 1. Generación de datos simulados: Se crea un conjunto de 5 objetos con pesos 
# y valores aleatorios.

# 2. Implementación de algoritmos: Se implementan diez soluciones al problema 
# de la mochila.

# 3. Medición de tiempos: Se mide el tiempo de ejecución para cada algoritmo y se 
# registra el valor máximo obtenido.

# 4. Graficación: Se grafican los tiempos de ejecución y los resultados de cada 
# algoritmo para una comparación visual.

# Resultados:
# 1. Después de ejecutar este código, se obtendrán dos gráficos:
# 2. Tiempo de ejecución: Mostrará cuánto tiempo tomó cada algoritmo para 
# resolver el problema de la mochila.
# Valor máximo obtenido: Comparará los valores máximos obtenidos por cada 
# algoritmo, mostrando cuál fue más eficiente en términos de la solución 
# encontrada.

# Estos gráficos proporcionarán una visión clara de la eficiencia y efectividad 
# de cada enfoque.
#==============================================================================

# Cargar librerías
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations
import pulp  # Para programación lineal
from deap import base, creator, tools, algorithms  # Para Algoritmo Genético

# Paso 1: Generar datos simulados con 5 artículos
np.random.seed(42)
num_items = 5
weights = np.random.randint(1, 100, size=num_items)
values = np.random.randint(10, 500, size=num_items)
capacity = 500

# Función para medir el tiempo de ejecución de los algoritmos
def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start

# Función para verificar la validez de la solución
def is_valid_solution(selected_items, weights, capacity):
    total_weight = sum(weights[i] for i in selected_items)
    return total_weight <= capacity

# Función para obtener los elementos seleccionados a partir de la solución
def get_selected_items(solution, weights):
    selected_items = []
    for i in range(len(solution)):
        if solution[i] == 1:
            selected_items.append(i)
    return selected_items

# Paso 2: Implementación de los algoritmos

# 1. Algoritmo de Fuerza Bruta
def brute_force_knapsack(weights, values, capacity):
    n = len(weights)
    max_value = 0
    best_combination = []
    for r in range(n+1):
        for comb in combinations(range(n), r):
            total_weight = sum(weights[i] for i in comb)
            total_value = sum(values[i] for i in comb)
            if total_weight <= capacity and total_value > max_value:
                max_value = total_value
                best_combination = comb
    return max_value, best_combination

# 2. Programación Dinámica
def dynamic_programming_knapsack(weights, values, capacity):
    n = len(weights)
    dp = np.zeros((n+1, capacity+1))
    for i in range(1, n+1):
        for w in range(capacity+1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)
            w -= weights[i-1]
    return dp[n][capacity], selected_items

# 3. Algoritmo Voraz (Greedy)
def greedy_knapsack(weights, values, capacity):
    n = len(weights)
    index = list(range(n))
    ratio = [v/w for v, w in zip(values, weights)]
    index.sort(key=lambda i: ratio[i], reverse=True)
    max_value = 0
    total_weight = 0
    selected_items = []
    for i in index:
        if total_weight + weights[i] <= capacity:
            total_weight += weights[i]
            max_value += values[i]
            selected_items.append(i)
    return max_value, selected_items

# 4. Algoritmo de Backtracking
def backtracking_knapsack(weights, values, capacity, n, total_value=0, current_weight=0):
    if n == 0 or capacity == 0:
        return total_value, []
    if weights[n-1] > capacity:
        return backtracking_knapsack(weights, values, capacity, n-1, total_value, current_weight)
    else:
        include_value, include_items = backtracking_knapsack(weights, values, capacity-weights[n-1], n-1, total_value+values[n-1], current_weight+weights[n-1])
        exclude_value, exclude_items = backtracking_knapsack(weights, values, capacity, n-1, total_value, current_weight)
        if include_value > exclude_value:
            return include_value, include_items + [n-1]
        else:
            return exclude_value, exclude_items

# 5. Branch and Bound
class Node:
    def __init__(self, level, value, weight, bound, items):
        self.level = level
        self.value = value
        self.weight = weight
        self.bound = bound
        self.items = items

def bound(u, n, capacity, weights, values):
    if u.weight >= capacity:
        return 0
    else:
        result = u.value
        j = u.level + 1
        total_weight = u.weight
        while j < n and total_weight + weights[j] <= capacity:
            total_weight += weights[j]
            result += values[j]
            j += 1
        if j < n:
            result += (capacity - total_weight) * values[j] / weights[j]
        return result

def branch_and_bound_knapsack(weights, values, capacity):
    n = len(weights)
    Q = []
    u = Node(-1, 0, 0, 0, [])
    v = Node(-1, 0, 0, 0, [])
    u.bound = bound(u, n, capacity, weights, values)
    max_value = 0
    best_items = []
    Q.append(u)
    while Q:
        u = Q.pop(0)
        if u.level == -1:
            v.level = 0
        if u.level == n-1:
            continue
        v.level = u.level + 1
        v.weight = u.weight + weights[v.level]
        v.value = u.value + values[v.level]
        v.items = u.items + [v.level]
        if v.weight <= capacity and v.value > max_value:
            max_value = v.value
            best_items = v.items
        v.bound = bound(v, n, capacity, weights, values)
        if v.bound > max_value:
            Q.append(Node(v.level, v.value, v.weight, v.bound, v.items))
        v.weight = u.weight
        v.value = u.value
        v.items = u.items
        v.bound = bound(v, n, capacity, weights, values)
        if v.bound > max_value:
            Q.append(Node(v.level, v.value, v.weight, v.bound, v.items))
    return max_value, best_items

# 6. Programación Entera
def integer_programming_knapsack(weights, values, capacity):
    n = len(weights)
    prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")
    prob += pulp.lpSum([values[i] * x[i] for i in range(n)])
    prob += pulp.lpSum([weights[i] * x[i] for i in range(n)]) <= capacity
    prob.solve()
    selected_items = [i for i in range(n) if pulp.value(x[i]) == 1]
    return pulp.value(prob.objective), selected_items

# 7. Algoritmo Genético
def genetic_algorithm_knapsack(weights, values, capacity, population_size=50, generations=100):
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
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)
    
    best_individual = tools.selBest(population, 1)[0]
    selected_items = get_selected_items(best_individual, weights)
    return eval_knapsack(best_individual)[0], selected_items

# 8. Recocido Simulado
def simulated_annealing_knapsack(weights, values, capacity, temp=10000, cooling_rate=0.99, stopping_temp=0.1):
    current_solution = np.random.randint(2, size=len(weights))
    current_weight = sum(current_solution[i] * weights[i] for i in range(len(current_solution)))
    current_value = sum(current_solution[i] * values[i] for i in range(len(current_solution)))
    
    best_solution = np.copy(current_solution)
    best_value = current_value
    
    while temp > stopping_temp:
        new_solution = np.copy(current_solution)
        index = np.random.randint(len(weights))
        new_solution[index] = 1 - new_solution[index]
        new_weight = sum(new_solution[i] * weights[i] for i in range(len(new_solution)))
        new_value = sum(new_solution[i] * values[i] for i in range(len(new_solution)))
        
        if new_weight <= capacity and (new_value > current_value or np.exp((new_value - current_value) / temp) > np.random.rand()):
            current_solution = new_solution
            current_value = new_value
            current_weight = new_weight
            
            if current_value > best_value:
                best_solution = np.copy(current_solution)
                best_value = current_value
        
        temp *= cooling_rate
    
    selected_items = get_selected_items(best_solution, weights)
    return best_value, selected_items

# 9. Optimización con Colonias de Hormigas (ACO)
def aco_knapsack(weights, values, capacity, iterations=100, num_ants=50, alpha=1, beta=1, rho=0.5):
    n = len(weights)
    pheromone = np.ones(n)
    best_value = 0
    best_solution = np.zeros(n)
    
    for _ in range(iterations):
        for _ in range(num_ants):
            prob = pheromone**alpha * (values/weights)**beta
            prob = prob / prob.sum()
            solution = np.random.rand(n) < prob
            total_weight = sum(solution[i] * weights[i] for i in range(n))
            total_value = sum(solution[i] * values[i] for i in range(n))
            if total_weight <= capacity and total_value > best_value:
                best_value = total_value
                best_solution = solution
        
        pheromone *= (1 - rho)
        pheromone += best_solution
    
    selected_items = get_selected_items(best_solution, weights)
    return best_value, selected_items

# 10. Algoritmos Basados en Enjambre de Partículas (PSO)
def pso_knapsack(weights, values, capacity, population_size=50, iterations=100):
    n = len(weights)
    positions = np.random.randint(2, size=(population_size, n))
    velocities = np.random.randn(population_size, n)
    personal_best_positions = np.copy(positions)
    personal_best_values = np.array([sum(pos * values) if sum(pos * weights) <= capacity else 0 for pos in positions])
    global_best_position = personal_best_positions[np.argmax(personal_best_values)]
    
    for _ in range(iterations):
        for i in range(population_size):
            r1, r2 = np.random.rand(2)
            velocities[i] = 0.5 * velocities[i] + r1 * (personal_best_positions[i] - positions[i]) + r2 * (global_best_position - positions[i])
            positions[i] = np.where(np.random.rand(n) < 1 / (1 + np.exp(-velocities[i])), 1, 0)
            total_weight = sum(positions[i] * weights)
            total_value = sum(positions[i] * values)
            if total_weight <= capacity and total_value > personal_best_values[i]:
                personal_best_positions[i] = positions[i]
                personal_best_values[i] = total_value
                if total_value > sum(global_best_position * values):
                    global_best_position = positions[i]
    
    selected_items = get_selected_items(global_best_position, weights)
    return sum(global_best_position * values), selected_items

# Paso 3: Medición del tiempo de ejecución y obtención de resultados
knapsack_algorithms = {
    "Fuerza Bruta": brute_force_knapsack,
    "Programación Dinámica": dynamic_programming_knapsack,
    "Voraz": greedy_knapsack,
    "Backtracking": lambda w, v, c: backtracking_knapsack(w, v, c, len(w)),
    "Branch and Bound": branch_and_bound_knapsack,
    "Programación Entera": integer_programming_knapsack,
    "Algoritmo Genético": genetic_algorithm_knapsack,
    "Recocido Simulado": simulated_annealing_knapsack,
    "ACO": aco_knapsack,
    "PSO": pso_knapsack
}

results = []
times = []
validity = []

for name, algorithm in knapsack_algorithms.items():
    result, exec_time = measure_time(algorithm, weights, values, capacity)
    value, selected_items = result
    is_valid = is_valid_solution(selected_items, weights, capacity)
    validity.append(is_valid)
    results.append(value)
    times.append(exec_time)
    validity_status = "Válido" if is_valid else "No válido"
    print(f"{name}: Valor máximo = {value}, Tiempo de ejecución = {exec_time:.4f} segundos, Solución {validity_status}")

# Paso 4: Graficar los resultados
plt.figure(figsize=(12, 6))
plt.bar(knapsack_algorithms.keys(), times, color='skyblue')
plt.ylabel("Tiempo de ejecución (segundos)")
plt.title("Comparación del Tiempo de Ejecución de Diferentes Algoritmos para el Problema de la Mochila")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(knapsack_algorithms.keys(), results, color='lightgreen')
plt.ylabel("Valor Máximo Obtenido")
plt.title("Comparación del Valor Máximo Obtenido por Diferentes Algoritmos para el Problema de la Mochila")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
