
# -*- coding: utf-8 -*-
"""
Algoritmo Genético para el problema de la mochila con ítems repetibles.
@author: Jaime Aguilar Ortiz
"""
#==============================================================================
# PROBLEMA DE LA MOCHILA RESUELTO CON 10 ALGORITMOS Y 5 ARTÍCULOS CON
# MULTI ELECCIÓN
#==============================================================================

# Cargar librerías
import numpy as np

#=================== Parámetros del problema ======================
pesos = np.array([12, 2, 1, 4, 1])
valores = np.array([4, 2, 1, 10, 2])
capacidad = 15
num_items = len(pesos)

#=================== Parámetros del algoritmo genético ============
tamaño_poblacion = 100
tasa_cruce = 0.8
tasa_mutacion = 0.01
num_generaciones = 100

#=================== Función de aptitud ============================
def aptitud(individuo):
    peso_total = np.sum(individuo * pesos)
    valor_total = np.sum(individuo * valores)
    if peso_total <= capacidad:
        return valor_total
    else:
        return 0

#=================== Selección de padres ===========================
def seleccionar_padres(poblacion, aptitudes, num_padres):
    padres = np.empty((num_padres, poblacion.shape[1]), dtype=int)
    for i in range(num_padres):
        idx = np.argmax(aptitudes)
        padres[i, :] = poblacion[idx, :]
        aptitudes[idx] = -1
    return padres

#=================== Cruce genético ================================
def cruce(padres, tamaño_poblacion):
    descendencia = np.empty((tamaño_poblacion - padres.shape[0], padres.shape[1]), dtype=int)
    punto_cruce = np.uint8(padres.shape[1]/2)
    for k in range(descendencia.shape[0]):
        padre1_idx = k % padres.shape[0]
        padre2_idx = (k + 1) % padres.shape[0]
        descendencia[k, 0:punto_cruce] = padres[padre1_idx, 0:punto_cruce]
        descendencia[k, punto_cruce:] = padres[padre2_idx, punto_cruce:]
    return descendencia

#=================== Mutación genética =============================
def mutacion(descendencia):
    for i in range(descendencia.shape[0]):
        if np.random.uniform(0, 1) < tasa_mutacion:
            gen = np.random.randint(0, descendencia.shape[1])
            cambio = np.random.choice([-1, 1])
            descendencia[i, gen] = max(0, descendencia[i, gen] + cambio)
    return descendencia

#=================== Inicialización de la población ================
# Se asegura que los individuos no excedan capacidad ** AQUÍ FUE EL CAMBIO**
def generar_individuo_factible():
    individuo = np.zeros(num_items, dtype=int)
    peso_actual = 0
    while True:
        posibles = np.where(peso_actual + pesos <= capacidad)[0]
        if len(posibles) == 0:
            break
        item = np.random.choice(posibles)
        individuo[item] += 1
        peso_actual += pesos[item]
    return individuo

poblacion = np.array([generar_individuo_factible() for _ in range(tamaño_poblacion)])

#=================== Evolución del algoritmo =======================
for generacion in range(num_generaciones):
    aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
    padres = seleccionar_padres(poblacion.copy(), aptitudes.copy(), int(tasa_cruce * tamaño_poblacion))
    descendencia = cruce(padres, tamaño_poblacion)
    descendencia = mutacion(descendencia)
    poblacion[0:padres.shape[0], :] = padres
    poblacion[padres.shape[0]:, :] = descendencia

#=================== Resultado final ===============================
aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
mejor_solucion = poblacion[np.argmax(aptitudes)]
print("Mejor solución (ítems y cantidades):", mejor_solucion)
print("Peso total:", np.sum(mejor_solucion * pesos))
print("Valor total:", np.max(aptitudes))
