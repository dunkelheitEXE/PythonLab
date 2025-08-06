# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:29:04 2024

@author: Jaime
"""
#==============================================================================
# # Ejemplo 3 del diseño de un circuito eléctrico óptimo
# El código implementa un algoritmo genético multiobjetivo utilizando la 
# biblioteca DEAP para optimizar el diseño de circuitos electrónicos, 
# maximizando el rendimiento y minimizando el costo bajo la restricción de un 
# costo máximo permitido. Cada circuito se modela como un individuo compuesto 
# por 10 componentes cuyos valores se generan aleatoriamente entre 0 y 1. A 
# cada individuo se le asigna una aptitud basada en dos objetivos: el costo 
# total (calculado como la suma de los componentes multiplicada por 10) y el 
# rendimiento (calculado como el producto de los componentes incrementados en 
# uno), penalizando el rendimiento a cero si se supera el costo máximo. En cada 
# generación, se aplica un esquema evolutivo que incluye elitismo (preservando 
# los 5 mejores individuos), cruce uniforme con probabilidad del 70 %, mutación 
# gaussiana con mayor dispersión y probabilidad del 30 %, y selección por torneo 
# para mantener la presión selectiva. Además, se introducen 100 nuevos individuos 
# aleatorios en cada generación, lo que fortalece la diversidad genética de la 
# población y mitiga el riesgo de convergencia prematura. El algoritmo se 
# ejecuta por 500 generaciones, y al final se realiza un análisis gráfico y 
# estadístico de los resultados mediante diagramas de dispersión y de 
# frecuencia, con el objetivo de visualizar la distribución de soluciones en el 
# espacio de costo-rendimiento y evaluar la calidad del conjunto final de 
# circuitos obtenidos.
#==============================================================================

# Cargar librerías
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Parámetros del problema
NUM_COMPONENTES = 10  # Número de componentes en el circuito
NUM_INDIVIDUOS = 300  # Tamaño de la población
NUM_GENERACIONES = 500  # Número de generaciones
COSTO_MAXIMO = 100  # Costo máximo permitido para el circuito
ELITE_SIZE = 5  # Número de mejores individuos a preservar (ajustado)
NUEVOS_INDIVIDUOS = 100  # Número de individuos nuevos que se introducen en cada generación

# Crear las clases de Fitness y Individuo
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimizar costo, maximizar rendimiento
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Crear un individuo aleatorio (un circuito)
def crear_individuo():
    return creator.Individual([random.uniform(0, 1) for _ in range(NUM_COMPONENTES)])

# Evaluar el individuo (circuito)
def evaluar_individuo(individuo):
    costo = sum(individuo) * 10  # Supongamos que cada componente tiene un costo asociado
    rendimiento = np.prod([1 + x for x in individuo])  # Ejemplo de una función de rendimiento
    if costo > COSTO_MAXIMO:
        rendimiento = 0  # Penalización si el costo excede el máximo permitido
    costo = max(costo, 0)  # Asegurarse de que el costo nunca sea negativo
    return costo, rendimiento

# Configuración de la caja de herramientas
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initIterate, creator.Individual, crear_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxUniform, indpb=0.5)  # Usar cruce uniforme para mayor diversidad
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)  # Aumentar la tasa de mutación y sigma
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluar_individuo)

# Función principal
def main():
    random.seed(42)
    
    # Crear población inicial
    pop = toolbox.population(n=NUM_INDIVIDUOS)
    
    # Algoritmo evolutivo con elitismo y aumento de diversidad
    for gen in range(NUM_GENERACIONES):
        # Evaluar a los individuos en la población
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Selección de los mejores individuos (elitismo)
        elite = tools.selBest(pop, ELITE_SIZE)
        
        # Selección de los individuos para la próxima generación
        offspring = toolbox.select(pop, len(pop) - ELITE_SIZE - NUEVOS_INDIVIDUOS)
        
        # Aplicar cruce y mutación
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # Probabilidad de cruce
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.3:  # Probabilidad de mutación
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluar fitness de los nuevos individuos
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        
        # Introducir nuevos individuos aleatorios para aumentar la diversidad
        nuevos_individuos = [toolbox.individual() for _ in range(NUEVOS_INDIVIDUOS)]
        for ind in nuevos_individuos:
            ind.fitness.values = toolbox.evaluate(ind)

        # Reemplazar la población con la nueva generación + elitismo + nuevos individuos
        pop[:] = elite + offspring + nuevos_individuos
    
    # Obtener los resultados
    costos = [ind.fitness.values[0] for ind in pop]
    rendimientos = [ind.fitness.values[1] for ind in pop]

    return pop, costos, rendimientos

# Ejecutar el algoritmo
pop, costos, rendimientos = main()

# Análisis y gráficos de los resultados
def analizar_resultados(costos, rendimientos):
    # Gráfico de dispersión de costo vs rendimiento
    plt.figure(figsize=(10, 6))
    plt.scatter(costos, rendimientos, c='blue', marker='o')
    plt.title('Costo vs Rendimiento de los Circuitos')
    plt.xlabel('Costo')
    plt.ylabel('Rendimiento')
    plt.grid(True)
    plt.show()

    # Indicadores
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

# Ejecutar el análisis de resultados
analizar_resultados(costos, rendimientos)
