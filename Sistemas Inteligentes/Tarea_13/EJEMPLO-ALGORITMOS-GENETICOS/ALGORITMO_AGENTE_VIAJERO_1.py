# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:36:17 2024

@author: Jaime
"""

#==============================================================================
# Ejemplo 1 del agente viajero
# El código presentado implementa un algoritmo genético para resolver el 
# problema del agente viajero (TSP, por sus siglas en inglés), cuya finalidad 
# es encontrar la ruta más corta que visita una serie de ciudades exactamente 
# una vez y regresa al punto de partida. Para ello, se comienza generando una 
# matriz de distancias simétrica entre ciudades con valores aleatorios, lo cual 
# simula un entorno realista. A continuación, se crea una población inicial de 
# posibles rutas (cromosomas), cada una representada por una permutación 
# aleatoria de las ciudades. La aptitud de cada ruta se evalúa mediante la 
# distancia total recorrida, y se seleccionan padres para reproducción 
# utilizando un mecanismo de selección por torneo, que favorece las rutas más 
# cortas. El proceso de cruce (crossover) genera nuevas rutas combinando 
# segmentos de los padres, mientras que la mutación introduce variabilidad 
# aleatoria intercambiando ciudades dentro de una ruta con una probabilidad 
# determinada. Este ciclo de selección, cruce y mutación se repite durante un 
# número fijo de generaciones, manteniendo y actualizando la mejor solución 
# encontrada en cada iteración. Al finalizar, se imprime la mejor ruta hallada 
# y su correspondiente distancia mínima, evidenciando la capacidad del 
# algoritmo genético para aproximar soluciones óptimas en problemas 
# combinatorios complejos como el TSP.
#==============================================================================

# Cargar librerías
import random
import numpy as np

# Generación de distancia aleatoria entre ciudades
# Esta función denominada generar_matriz_distancias crea una matriz cuadrada de 
# tamaño num_ciudades × num_ciudades cuyos valores representan distancias 
# aleatorias entre pares de ciudades. Inicialmente, se generan números 
# aleatorios entre 0 y 100 para cada par de ciudades, pero como las distancias 
# entre dos ciudades deben ser iguales en ambos sentidos, se hace que la matriz 
# sea simétrica promediándola con su transpuesta. Posteriormente, se asigna un 
# valor de cero a la diagonal principal, ya que la distancia de una ciudad 
# consigo misma debe ser nula. El resultado es una matriz de distancias que 
# modela de manera realista el entorno para el problema del agente viajero.
def generar_matriz_distancias(num_ciudades):
    matriz = np.random.rand(num_ciudades, num_ciudades) * 100
    matriz = (matriz + matriz.T) / 2  # Hacer la matriz simétrica
    np.fill_diagonal(matriz, 0)
    return matriz

# Crear una ruta inicial (cromosoma)
# Esta función denominada crear_ruta genera una ruta inicial aleatoria, que en 
# el contexto de un algoritmo genético representa un cromosoma. Para ello, se 
# crea una lista con los identificadores numéricos de todas las ciudades, 
# utilizando range(num_ciudades), lo que garantiza que cada ciudad esté incluida 
# una única vez. Luego, esta lista se desordena aleatoriamente mediante la 
# función random.shuffle, produciendo así una permutación válida de las ciudades 
# que representa una posible solución al problema del agente viajero. La función 
# retorna esta ruta aleatoria, que sirve como punto de partida para el proceso 
# evolutivo del algoritmo genético.
def crear_ruta(num_ciudades):
    ruta = list(range(num_ciudades))
    random.shuffle(ruta)
    return ruta

# Evaluar la aptitud de una ruta (cromosoma)
# La función evaluar_ruta calcula la aptitud de una ruta (o cromosoma) midiendo 
# la distancia total recorrida al visitar todas las ciudades en el orden 
# especificado por la lista ruta, utilizando la matriz_distancias. Se suma la 
# distancia entre cada par consecutivo de ciudades, y al final se añade la 
# distancia desde la última ciudad de la ruta de vuelta a la ciudad de origen, 
# cerrando así el ciclo. Esta suma total representa el costo del recorrido y es 
# utilizada por el algoritmo genético como criterio de evaluación: cuanto menor 
# sea esta distancia, mayor será la aptitud de la ruta como solución al 
# problema del agente viajero.
def evaluar_ruta(ruta, matriz_distancias):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += matriz_distancias[ruta[i], ruta[i+1]]
    distancia_total += matriz_distancias[ruta[-1], ruta[0]]  # Volver al inicio
    return distancia_total

# Selección por torneo
# La función seleccion_torneo implementa el método de selección por torneo, una 
# técnica común en algoritmos genéticos para elegir individuos (en este caso, 
# rutas) que participarán en la reproducción. La función selecciona 
# aleatoriamente un subconjunto de individuos de la población (tamaño 
# determinado por tamano_torneo), junto con sus respectivas aptitudes, y luego 
# los ordena en función de su valor de aptitud (distancia total recorrida). El 
# individuo con menor distancia —es decir, con mayor aptitud— es elegido como 
# ganador del torneo y retornado para ser usado como progenitor en la siguiente 
# generación. Este método es adecuado en este caso porque balancea la 
# exploración y explotación, favoreciendo la selección de rutas de alta calidad 
# sin eliminar completamente la posibilidad de elegir soluciones peores, lo que 
# ayuda a mantener la diversidad genética y evitar la convergencia prematura 
# del algoritmo.
def seleccion_torneo(poblacion, aptitudes, tamano_torneo=3):
    seleccionados = random.sample(list(zip(poblacion, aptitudes)), tamano_torneo)
    seleccionados.sort(key=lambda x: x[1])
    return seleccionados[0][0]

# Cruce de rutas (crossover)
# La función crossover implementa un operador de cruce (o recombinación) entre 
# dos rutas (denominadas padre1 y padre2) para generar un nuevo individuo 
# (o hijo) que combine características de ambos progenitores. Primero, se 
# selecciona aleatoriamente un segmento de la ruta del padre1, que se copia 
# directamente al hijo en las posiciones correspondientes. Luego, se recorre la 
# ruta del padre2 y se insertan las ciudades que aún no están en el hijo, 
# respetando el orden en que aparecen en el padre2 y colocándolas en los 
# espacios vacíos (marcados inicialmente con -1). Este método garantiza que el 
# hijo sea una permutación válida de ciudades, es decir, que incluya todas las 
# ciudades sin repeticiones, manteniendo así la viabilidad de la solución en el 
# contexto del problema del agente viajero.
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
# La función mutacion introduce variabilidad genética en una ruta (o cromosoma)
# mediante un proceso aleatorio de intercambio de ciudades. Para cada posición 
# de la lista ruta, se genera un número aleatorio y, si este es menor que la 
# tasa_mutacion especificada, se selecciona al azar otra posición j y se 
# intercambian los elementos de las posiciones i y j. Este proceso permite 
# modificar ligeramente la estructura de la ruta sin alterar su validez como 
# permutación, evitando que el algoritmo genético se estanque en soluciones 
# subóptimas y contribuyendo a explorar nuevas regiones del espacio de 
# soluciones, lo cual es esencial para mantener la diversidad poblacional a lo 
# largo de las generaciones.
def mutacion(ruta, tasa_mutacion):
    for i in range(len(ruta)):
        if random.random() < tasa_mutacion:
            j = random.randint(0, len(ruta) - 1)
            ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta

# Algoritmo genético para el problema del agente viajero
# La función algoritmo_genetico implementa un procedimiento evolutivo completo 
# para resolver el problema del agente viajero (TSP), optimizando la ruta más 
# corta que visita todas las ciudades una vez y regresa al punto de partida. 
# Inicialmente, se genera una matriz de distancias simétrica aleatoria entre 
# ciudades y se construye una población inicial de rutas aleatorias. A lo largo 
# de un número determinado de generaciones, el algoritmo evalúa la aptitud de 
# cada ruta (es decir, su distancia total), selecciona padres mediante torneos, 
# genera nuevos individuos usando cruce y mutación, y reemplaza la población 
# anterior con la nueva. Durante el proceso, se lleva un registro de la mejor 
# solución encontrada hasta el momento, tanto en términos de ruta como de 
# distancia. Al final de todas las generaciones, se retorna esta mejor ruta 
# junto con su distancia mínima, reflejando la capacidad adaptativa del 
# algoritmo genético para aproximarse a soluciones óptimas en problemas 
# combinatorios de alta complejidad.
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
    
    return mejor_ruta, mejor_distancia

# Parámetros
num_ciudades = 20
tamano_poblacion = 100
tasa_mutacion = 0.01
num_generaciones = 500

# Ejecución del algoritmo genético
mejor_ruta, mejor_distancia = algoritmo_genetico(num_ciudades, tamano_poblacion, tasa_mutacion, num_generaciones)
print("Mejor ruta encontrada:", mejor_ruta)
print("Distancia total de la mejor ruta:", mejor_distancia)
