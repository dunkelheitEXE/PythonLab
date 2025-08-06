# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:40:11 2024

@author: Jaime
"""

#==============================================================================
# # Ejemplo 2 del agente viajero
# Anexa la rura al ejemplo 1
#==============================================================================

# Cargar librerías
import random
import numpy as np
import matplotlib.pyplot as plt

# Generación de distancia aleatoria entre ciudades
def generar_matriz_distancias(num_ciudades):
    matriz = np.random.rand(num_ciudades, num_ciudades) * 100
    matriz = (matriz + matriz.T) / 2  # Hacer la matriz simétrica
    np.fill_diagonal(matriz, 0)
    return matriz

# Crear una ruta inicial (cromosoma)
def crear_ruta(num_ciudades):
    ruta = list(range(num_ciudades))
    random.shuffle(ruta)
    return ruta

# Evaluar la aptitud de una ruta (cromosoma)
def evaluar_ruta(ruta, matriz_distancias):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += matriz_distancias[ruta[i], ruta[i+1]]
    distancia_total += matriz_distancias[ruta[-1], ruta[0]]  # Volver al inicio
    return distancia_total

# Selección por torneo
def seleccion_torneo(poblacion, aptitudes, tamano_torneo=3):
    seleccionados = random.sample(list(zip(poblacion, aptitudes)), tamano_torneo)
    seleccionados.sort(key=lambda x: x[1])
    return seleccionados[0][0]

# Cruce de rutas (crossover)
def crossover(padre1, padre2):
    inicio = random.randint(0, len(padre1) - 2)
    fin = random.randint(inicio + 1, len(padre1) - 1)
    
    hijo = [-1] * len(padre1)
    hijo[inicio:fin] = padre1[inicio:fin]
    
    for ciudad in padre2:
        if ciudad not in hijo:
            for i in range(len(hijo)):
                if hijo[i] == -1:
                    hijo[i] = ciudad
                    break
    return hijo

# Mutación de una ruta
def mutacion(ruta, tasa_mutacion):
    for i in range(len(ruta)):
        if random.random() < tasa_mutacion:
            j = random.randint(0, len(ruta) - 1)
            ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta

# Algoritmo genético para el problema del agente viajero
def algoritmo_genetico(num_ciudades, tamano_poblacion, tasa_mutacion, num_generaciones):
    matriz_distancias = generar_matriz_distancias(num_ciudades)
    poblacion = [crear_ruta(num_ciudades) for _ in range(tamano_poblacion)]
    
    mejor_ruta = None
    mejor_distancia = float('inf')
    
    for generacion in range(num_generaciones):
        aptitudes = [evaluar_ruta(ruta, matriz_distancias) for ruta in poblacion]
        
        nueva_poblacion = []
        for _ in range(tamano_poblacion):
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            hijo = crossover(padre1, padre2)
            hijo = mutacion(hijo, tasa_mutacion)
            nueva_poblacion.append(hijo)
        
        poblacion = nueva_poblacion
        
        min_distancia = min(aptitudes)
        if min_distancia < mejor_distancia:
            mejor_distancia = min_distancia
            mejor_ruta = poblacion[aptitudes.index(min_distancia)]
        
        print(f"Generación {generacion + 1}: Mejor distancia = {mejor_distancia}")
    
    return mejor_ruta, mejor_distancia, matriz_distancias

# Visualización de la mejor ruta
def visualizar_ruta(ruta, matriz_distancias):
    # Coordenadas aleatorias para las ciudades
    coordenadas = np.random.rand(len(ruta), 2)
    
    plt.figure(figsize=(10, 6))
    
    # Dibujar las ciudades
    plt.scatter(coordenadas[:, 0], coordenadas[:, 1], c='red')
    
    # Dibujar las conexiones entre las ciudades
    for i in range(len(ruta)):
        ciudad_inicio = ruta[i]
        ciudad_fin = ruta[(i + 1) % len(ruta)]
        plt.plot([coordenadas[ciudad_inicio, 0], coordenadas[ciudad_fin, 0]], 
                 [coordenadas[ciudad_inicio, 1], coordenadas[ciudad_fin, 1]], 'b-')
    
    # Etiquetas de las ciudades
    for i, coord in enumerate(coordenadas):
        plt.text(coord[0], coord[1], f'{i}', fontsize=12, ha='right')
    
    plt.title("Mejor ruta encontrada")
    plt.show()

# Parámetros
num_ciudades = 20
tamano_poblacion = 100
tasa_mutacion = 0.01
num_generaciones = 500

# Ejecución del algoritmo genético
mejor_ruta, mejor_distancia, matriz_distancias = algoritmo_genetico(num_ciudades, tamano_poblacion, tasa_mutacion, num_generaciones)

# Imprimir resultados
print("Mejor ruta encontrada:", mejor_ruta)
print("Distancia total de la mejor ruta:", mejor_distancia)

# Visualizar la mejor ruta
visualizar_ruta(mejor_ruta, matriz_distancias)
