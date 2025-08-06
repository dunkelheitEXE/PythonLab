# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 19:59:52 2025

@author: Asus.S510UNR
"""

"""
Optimización evolutiva de un circuito eléctrico: costo mínimo y rendimiento máximo.
Modificaciones: restricción inferior a los componentes, penalización por costos bajos y validación final.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Parámetros del problema
NUM_COMPONENTES = 10        # Número de componentes en el circuito
NUM_INDIVIDUOS = 300        # Tamaño de la población
NUM_GENERACIONES = 500      # Número de generaciones
COSTO_MAXIMO = 100          # Costo máximo permitido para el circuito
UMBRAL_MINIMO_COSTO = 1.0   # Umbral mínimo para considerar válido un circuito

# Crear las clases de fitness y individuo
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimizar costo, maximizar rendimiento
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Crear un individuo aleatorio (valores entre 0.1 y 1 en lugar de 0 a 1)
def crear_individuo():
    return creator.Individual([random.uniform(0.1, 1.0) for _ in range(NUM_COMPONENTES)])

# Evaluar el individuo (circuito)
def evaluar_individuo(individuo):
    costo = sum(individuo) * 10
    rendimiento = np.prod([1 + x for x in individuo])

    # Penalización por costo fuera del rango permitido
    if costo > COSTO_MAXIMO or costo < UMBRAL_MINIMO_COSTO:
        rendimiento = 0

    # Seguridad adicional
    costo = max(costo, 0)
    return costo, rendimiento

# Configuración de la caja de herramientas
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.1, 1.0)
toolbox.register("individual", tools.initIterate, creator.Individual, crear_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluar_individuo)

# Función principal
def main():
    random.seed(42)
    
    # Crear población inicial
    pop = toolbox.population(n=NUM_INDIVIDUOS)
    
    # Ejecutar algoritmo evolutivo
    algorithms.eaMuPlusLambda(pop, toolbox, mu=NUM_INDIVIDUOS, lambda_=NUM_INDIVIDUOS,
                              cxpb=0.5, mutpb=0.2, ngen=NUM_GENERACIONES,
                              stats=None, halloffame=None, verbose=True)
    
    # Obtener resultados válidos filtrando circuitos triviales
    costos = []
    rendimientos = []
    for ind in pop:
        suma_componentes = sum(ind)
        if suma_componentes >= (UMBRAL_MINIMO_COSTO / 10.0):  # Costo mínimo expresado en términos de suma
            costos.append(ind.fitness.values[0])
            rendimientos.append(ind.fitness.values[1])

    return pop, costos, rendimientos

# Ejecutar algoritmo
pop, costos, rendimientos = main()

# Análisis y gráficos de resultados
def analizar_resultados(costos, rendimientos):
    # Gráfico de dispersión costo vs rendimiento
    plt.figure(figsize=(10, 6))
    plt.scatter(costos, rendimientos, c='blue', marker='o')
    plt.title('Costo vs Rendimiento de los Circuitos')
    plt.xlabel('Costo')
    plt.ylabel('Rendimiento')
    plt.grid(True)
    plt.show()

    # Indicadores estadísticos
    mejor_costo = min(costos)
    mejor_rendimiento = max(rendimientos)
    promedio_costo = np.mean(costos)
    promedio_rendimiento = np.mean(rendimientos)

    print(f"Mejor Costo: {mejor_costo}")
    print(f"Mejor Rendimiento: {mejor_rendimiento}")
    print(f"Costo Promedio: {promedio_costo}")
    print(f"Rendimiento Promedio: {promedio_rendimiento}")

    # Histograma de los costos
    plt.figure(figsize=(10, 6))
    plt.hist(costos, bins=20, color='orange', edgecolor='black')
    plt.title('Distribución de Costos de los Circuitos')
    plt.xlabel('Costo')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

    # Histograma de los rendimientos
    plt.figure(figsize=(10, 6))
    plt.hist(rendimientos, bins=20, color='green', edgecolor='black')
    plt.title('Distribución de Rendimientos de los Circuitos')
    plt.xlabel('Rendimiento')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

# Ejecutar análisis final
analizar_resultados(costos, rendimientos)
