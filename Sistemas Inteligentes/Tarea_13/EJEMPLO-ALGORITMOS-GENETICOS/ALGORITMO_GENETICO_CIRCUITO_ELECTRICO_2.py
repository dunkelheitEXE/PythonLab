# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:15:48 2024

@author: Jaime
"""

#==============================================================================
# Ejemplo 2 del diseño de un circuito eléctrico óptimo
# El código implementa un algoritmo genético multiobjetivo utilizando la 
# biblioteca DEAP, con el fin de optimizar simultáneamente el costo y el 
# rendimiento de un circuito electrónico compuesto por 10 componentes. Cada 
# circuito se representa como un individuo, codificado como una lista de valores 
# reales entre 0 y 1, y evaluado con una función de aptitud que calcula el costo 
# total (la suma de los componentes multiplicada por 10) y el rendimiento (el 
# producto de los componentes aumentados en 1), penalizando con rendimiento nulo 
# si se excede un costo máximo de 100. La población inicial se genera de forma 
# aleatoria, y en cada generación se aplican operadores genéticos: selección 
# por torneo, cruce uniforme (con 70 % de probabilidad), y mutación gaussiana 
# (con 30 % de probabilidad). Aunque el parámetro ELITE_SIZE está ajustado a 
# cero, el código incorpora una estructura que permitiría la preservación de 
# los mejores individuos, característica conocida como elitismo. El proceso 
# evolutivo se ejecuta durante 500 generaciones, tras lo cual se recogen los 
# valores finales de costo y rendimiento de la población. Finalmente, se realiza 
# un análisis visual mediante gráficos de dispersión y de histograma para 
# examinar la calidad de las soluciones obtenidas, así como un resumen 
# estadístico de los indicadores más relevantes. En conjunto, este algoritmo 
# permite explorar el espacio de diseño de circuitos para identificar 
# configuraciones que logren un equilibrio entre bajo costo y alto rendimiento.
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
ELITE_SIZE = 0  # Número de mejores individuos a preservar (ajustado)

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
toolbox.register("mate", tools.cxUniform, indpb=0.5)  # Usar cruce uniforme
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluar_individuo)

# Función principal
def main():
    random.seed(42)
    
    # Crear población inicial
    pop = toolbox.population(n=NUM_INDIVIDUOS)
    
    # Algoritmo evolutivo con elitismo
    for gen in range(NUM_GENERACIONES):
        # Evaluar a los individuos en la población
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Selección de los mejores individuos (elitismo)
        elite = tools.selBest(pop, ELITE_SIZE)
        
        # Selección de los individuos para la próxima generación
        offspring = toolbox.select(pop, len(pop) - ELITE_SIZE)
        
        # Aplicar cruce y mutación
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # Aumentar probabilidad de cruce
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.3:  # Aumentar probabilidad de mutación
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluar fitness de los nuevos individuos
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        
        # Reemplazar la población con la nueva generación + elitismo
        pop[:] = elite + offspring
    
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
