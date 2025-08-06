# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:41:01 2024

@author: Jaime
"""
#==============================================================================
# Ejemplo 1 del diseño de un circuito eléctrico óptimo
# El código presentado implementa un algoritmo evolutivo multiobjetivo utilizando 
# la biblioteca DEAP (Distributed Evolutionary Algorithms in Python) para 
# diseñar circuitos electrónicos optimizados simultáneamente en términos de 
# mínimo costo y máximo rendimiento. Inicialmente, se definen los parámetros del 
# problema, como el número de componentes en cada circuito (individuo), el 
# tamaño de la población, las generaciones a simular y el costo máximo permitido. 
# Se crea una clase de individuo que hereda de list y asocia un esquema de 
# aptitud con dos objetivos opuestos (minimizar el costo y maximizar el 
# rendimiento). Cada individuo se genera con valores aleatorios entre 0 y 1 
# para cada componente, representando configuraciones posibles del circuito. El 
# rendimiento se calcula como el producto de transformaciones de cada componente, 
# mientras que el costo es la suma ponderada de sus valores; si el costo excede 
# un umbral, se penaliza el rendimiento a cero. Luego, se configuran los 
# operadores genéticos (cruce, mutación, selección) utilizando cxTwoPoint, 
# mutGaussian y el algoritmo NSGA-II (selNSGA2) para manejar el conflicto entre 
# objetivos. El ciclo evolutivo se ejecuta con el algoritmo eaMuPlusLambda, que 
# implementa una estrategia elitista de supervivencia y reproducción. Al 
# finalizar, se extraen las métricas de los individuos resultantes y se realiza 
# un análisis visual y estadístico mediante gráficos de dispersión y histogramas 
# para evaluar la calidad de las soluciones encontradas. De este modo, el 
# código permite explorar un espacio de diseño complejo y obtener un conjunto 
# diverso de circuitos eficientes y económicamente viables.
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

# Crear las clases de fitness y individuo
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
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluar_individuo)

# Función principal
def main():
    random.seed(42)
    
    # Crear población inicial
    pop = toolbox.population(n=NUM_INDIVIDUOS)
    
    # Algoritmo evolutivo
    algorithms.eaMuPlusLambda(pop, toolbox, mu=NUM_INDIVIDUOS, lambda_=NUM_INDIVIDUOS,
                              cxpb=0.5, mutpb=0.2, ngen=NUM_GENERACIONES,
                              stats=None, halloffame=None, verbose=True)
    
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
