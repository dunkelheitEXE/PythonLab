# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:28:21 2025

@author: Asus.S510UNR
"""

#==============================================================================
# ALGORITMO GENÉTICO MEJORADO PARA EL PROBLEMA DE LA MOCHILA CON OPCIÓN BINARIA
#==============================================================================

# Cargar librerías
import numpy as np

#-------------------------------
# 1. Parámetros y datos
#-------------------------------
pesos = np.array([12, 2, 1, 4, 1])
valores = np.array([4, 2, 1, 12, 2])
capacidad = 15

tamaño_poblacion = 100
tasa_cruce = 0.8
tasa_mutacion = 0.05
num_generaciones = 100

#-------------------------------
# 2. Función de aptitud
#-------------------------------
def aptitud(individuo):
    peso_total = np.sum(individuo * pesos)
    valor_total = np.sum(individuo * valores)
    return valor_total if peso_total <= capacidad else 0

#-------------------------------
# 3. Selección por torneo binario
#-------------------------------
def seleccionar_padres(poblacion, aptitudes, num_padres):
    padres = []
    for _ in range(num_padres):
        i, j = np.random.choice(range(len(poblacion)), size=2, replace=False)
        ganador = poblacion[i] if aptitudes[i] > aptitudes[j] else poblacion[j]
        padres.append(ganador.copy())
    return np.array(padres)

#-------------------------------
# 4. Cruce de un punto aleatorio
#-------------------------------
def cruce(padres, tamaño_poblacion):
    num_descendencia = tamaño_poblacion - len(padres)
    descendencia = np.empty((num_descendencia, padres.shape[1]), dtype=int)
    for k in range(num_descendencia):
        padre1 = padres[np.random.randint(len(padres))]
        padre2 = padres[np.random.randint(len(padres))]
        punto = np.random.randint(1, len(padre1))
        hijo = np.concatenate([padre1[:punto], padre2[punto:]])
        descendencia[k] = hijo
    return descendencia

#-------------------------------
# 5. Mutación
#-------------------------------
def mutacion(descendencia):
    for individuo in descendencia:
        for gen in range(len(individuo)):
            if np.random.rand() < tasa_mutacion:
                individuo[gen] = 1 - individuo[gen]
    return descendencia

#-------------------------------
# 6. Inicialización de población
#-------------------------------
poblacion = np.random.randint(2, size=(tamaño_poblacion, len(pesos)))

#-------------------------------
# 7. Evolución
#-------------------------------
for generacion in range(num_generaciones):
    aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
    padres = seleccionar_padres(poblacion, aptitudes, int(tasa_cruce * tamaño_poblacion))
    descendencia = cruce(padres, tamaño_poblacion)
    descendencia = mutacion(descendencia)
    poblacion[:len(padres)] = padres
    poblacion[len(padres):] = descendencia

    # Registro de progreso
    mejor = poblacion[np.argmax(aptitudes)]
    print(f"Generación {generacion + 1}: Mejor valor = {np.max(aptitudes)}, Solución = {mejor}")

#-------------------------------
# 8. Resultado final
#-------------------------------
aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
mejor_solucion = poblacion[np.argmax(aptitudes)]
print("=============================================")
print("Mejor solución final:", mejor_solucion)
print("Valor total:", np.max(aptitudes))
print("Peso total:", np.sum(mejor_solucion * pesos))
