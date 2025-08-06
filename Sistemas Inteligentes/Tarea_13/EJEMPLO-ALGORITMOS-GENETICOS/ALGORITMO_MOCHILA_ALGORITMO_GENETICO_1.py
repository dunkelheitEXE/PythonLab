# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:48:15 2023

@author: Jaime Aguilar Ortiz
"""
#==============================================================================
# ALGORITMO DE LA MOCHILA CON OPCIÓN BINARIA

# El código proporcionado implementa un algoritmo genético para resolver el 
# problema de la mochila, donde se busca maximizar el valor de los ítems 
# seleccionados sin exceder la capacidad máxima permitida. Se explica 
# brevemente lo que hace cada función.
#==============================================================================

# Cargar librería
import numpy as np

# Datos del problema: 
# Peso y valor de los ítems
pesos = [12, 2, 1, 4, 1]
valores = [4, 2, 1, 12, 2]
capacidad = 15

# Parámetros del algoritmo genético
tamaño_poblacion = 100
tasa_cruce = 0.8
tasa_mutacion = 0.01
num_generaciones = 100

# Funciones auxiliares

# # 1. Función aptitud (individuo):
# Propósito: Calcula la "aptitud" (fitness) de un individuo en la población.
# Descripción: Esta función suma el peso total y el valor total de los ítems 
# seleccionados por el individuo. Si el peso total está dentro del límite 
# permitido (capacidad), la función devuelve el valor total como aptitud. 
# Si el peso excede la capacidad, devuelve 0, lo que indica que la solución es 
# inválida.
def aptitud(individuo):
    peso_total = np.sum(individuo * pesos)
    valor_total = np.sum(individuo * valores)
    if peso_total <= capacidad:
        return valor_total
    else:
        return 0

# 2. Función seleccionar_padres (poblacion, aptitudes, num_padres):
# Propósito: Selecciona a los individuos más aptos de la población para que 
# actúen como "padres" en la siguiente generación.
# Descripción: Esta función identifica a los individuos con la mayor aptitud en 
# la población y los selecciona para ser padres. La selección se realiza 
# hasta que se alcanza el número de padres deseado. Los individuos 
# seleccionados se almacenan en un arreglo y se retiran temporalmente de la 
# lista de aptitudes para evitar seleccionar el mismo individuo varias veces.

# Dentro del algoritmo genético presentado, la aptitud de un padre para ser 
# seleccionado se mide utilizando la función de aptitud (individuo). Esta función 
# evalúa cada individuo en la población de acuerdo con los siguientes pasos:
    
# Cálculo del peso total: Se multiplica el vector del individuo (que representa 
# la inclusión o exclusión de ítems en la mochila) por el vector de pesos de 
# los ítems, y se suma el resultado. Este valor representa el peso total de los 
# ítems seleccionados por ese individuo.

# Cálculo del valor total: Similar al cálculo del peso, se multiplica el 
# vector del individuo por el vector de valores de los ítems, y se suma el 
# resultado. Este valor representa el valor total de los ítems seleccionados 
# por ese individuo.

# Verificación de la restricción de capacidad: Se compara el peso total 
# calculado con la capacidad máxima de la mochila. Si el peso total es menor o 
# igual a la capacidad, la aptitud del individuo es igual al valor total de los 
# ítems seleccionados. Si el peso total excede la capacidad, la aptitud del 
# individuo es 0, indicando que la solución no es válida.

# Este valor de aptitud es lo que se utiliza para medir qué tan "bueno" es un 
# individuo en la población.

# Selección de padres:
# En la función seleccionar_padres (poblacion, aptitudes, num_padres), los 
# individuos con las mayores aptitudes tienen una mayor probabilidad de ser 
# seleccionados como padres. La función selecciona iterativamente a los 
# individuos con la mayor aptitud de la población, los añade al conjunto de 
# padres, y los retira temporalmente de la lista de aptitudes (asignándoles un 
# valor de -1) para evitar seleccionar al mismo individuo más de una vez. Este 
# proceso continúa hasta que se ha seleccionado el número deseado de padres.

####### corregir
def seleccionar_padres(poblacion, aptitudes, num_padres):
    padres = np.empty((num_padres, poblacion.shape[1]))
    for padre in range(num_padres):
        max_aptitud_idx = np.where(aptitudes == np.max(aptitudes))
        max_aptitud_idx = max_aptitud_idx[0][0]
        padres[padre, :] = poblacion[max_aptitud_idx, :]
        aptitudes[max_aptitud_idx] = -1
    return padres
###### corregir 

# 3. Función cruce (padres, tamaño_poblacion):
# Propósito: Genera una nueva descendencia a partir de los padres seleccionados 
# mediante un proceso de cruce (crossover).
# Descripción: Esta función crea descendencia al combinar genes de dos padres 
# seleccionados. La descendencia se genera tomando la mitad de los genes del 
# primer padre y la otra mitad del segundo padre. Este proceso se repite hasta 
# que se llena la población de la siguiente generación.

def cruce(padres, tamaño_poblacion):
    descendencia = np.empty((tamaño_poblacion - padres.shape[0], padres.shape[1]))
    punto_cruce = np.uint8(padres.shape[1]/2)
    for k in range(descendencia.shape[0]):
        # Índice del primer padre
        padre1_idx = k % padres.shape[0]
        # Índice del segundo padre
        padre2_idx = (k+1) % padres.shape[0]
        # Genes del primer padre
        descendencia[k, 0:punto_cruce] = padres[padre1_idx, 0:punto_cruce]
        # Genes del segundo padre
        descendencia[k, punto_cruce:] = padres[padre2_idx, punto_cruce:]
    return descendencia

# 4. Función mutacion (descendencia):
# Propósito: Aplica una mutación aleatoria a la descendencia para mantener la 
# diversidad genética en la población.
# Descripción: La función recorre cada individuo en la descendencia y, con una 
# probabilidad igual a la tasa de mutación, selecciona un gen al azar para 
# mutarlo. Si el gen seleccionado es 0, se cambia a 1 y viceversa. Esto ayuda 
# a explorar nuevas soluciones y evita que el algoritmo se estanque en óptimos 
# locales.
def mutacion(descendencia):
    for idx in range(descendencia.shape[0]):
        if np.random.uniform(0, 1) < tasa_mutacion:
            gen_mutado = np.random.randint(0, descendencia.shape[1])
            if descendencia[idx, gen_mutado] == 0:
                descendencia[idx, gen_mutado] = 1
            else:
                descendencia[idx, gen_mutado] = 0
    return descendencia

# 5. Inicialización
# Se crea una población inicial aleatoria, donde cada individuo 
# es una solución potencial representada por un vector de ceros y unos 
# (0: ítem no incluido, 1: ítem incluido).
poblacion = np.random.randint(2, size=(tamaño_poblacion, len(pesos)))

# 6. Evolución 
# Durante un número fijo de generaciones, el algoritmo realiza los siguientes 
# pasos:
    
# Evalúa la aptitud de cada individuo en la población.
# Selecciona los individuos más aptos como padres.
# Genera una nueva descendencia a través del cruce de los padres.
# Aplica mutación a la descendencia.
# Reemplaza parte de la población actual con la nueva descendencia, asegurando 
# que los padres más aptos se mantengan.

# Esta línea inicia un ciclo for que se repetirá tantas veces como se haya 
# definido en num_generaciones. Cada iteración representa una generación del 
# algoritmo genético, es decir, un ciclo de evolución en el que se evalúa la 
# población, se seleccionan los padres, se reproducen, se mutan y se forma una 
# nueva población.
for generacion in range(num_generaciones):
    
# Se evalúa la aptitud (fitness) de cada individuo (posible solución) de la 
# población. Para ello, se aplica la función aptitud fila por fila (eje 1) a la 
# matriz poblacion utilizando np.apply_along_axis. El resultado es un arreglo 
# unidimensional donde cada valor representa la calidad de la solución 
# correspondiente (por ejemplo, una distancia total si se trata del problema 
# del agente viajero, donde menor valor indica mayor aptitud).
    aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
    
# Se seleccionan los padres que participarán en la reproducción. La cantidad de 
# padres seleccionados se determina como un porcentaje del tamaño de la 
# población, definido por el parámetro tasa_cruce. La función seleccionar_padres 
# probablemente implementa un mecanismo de selección como torneo, ruleta o 
# selección elitista, favoreciendo a los individuos con mejor aptitud.
    padres = seleccionar_padres(poblacion, aptitudes, int(tasa_cruce*tamaño_poblacion))
    
# Se genera la descendencia aplicando un operador de cruce (crossover) a los 
# padres seleccionados. El objetivo de esta operación es recombinar la 
# información genética de los padres para formar nuevos individuos (hijos) que 
# hereden características de ambos. La cantidad de hijos generados busca 
# completar el tamaño de la población total, por lo que tamaño_poblacion indica 
# cuántos nuevos individuos deben generarse.    
    descendencia = cruce(padres, tamaño_poblacion)
    
# Una vez creada la descendencia, se aplica un operador de mutación para 
# introducir pequeñas alteraciones aleatorias en los individuos. Este paso es 
# crucial para mantener la diversidad genética y evitar que el algoritmo quede 
# atrapado en óptimos locales. La mutación puede consistir en intercambiar 
# elementos de una ruta, alterar genes, etc., dependiendo del problema.
    descendencia = mutacion(descendencia)

# Se actualiza la población incluyendo a los padres seleccionados directamente, 
# probablemente como parte de una estrategia elitista, asegurando que los mejores 
# individuos se conserven sin modificación en la siguiente generación. Se 
# sobrescriben las primeras posiciones de la matriz poblacion con los padres.
    poblacion[0:padres.shape[0], :] = padres
    
# Se completa la nueva generación insertando a los hijos generados mediante 
# cruce y mutación en las posiciones restantes de la población. De este modo, 
# la nueva población está compuesta tanto por una élite (los padres) como por 
# nuevas soluciones (la descendencia), promoviendo tanto la explotación como la 
# exploración del espacio de soluciones.
    poblacion[padres.shape[0]:, :] = descendencia

# 6. Resultado final 
# Al final de todas las generaciones, el código evalúa la aptitud de la 
# población final y selecciona la mejor solución (individuo con la mayor 
# aptitud). Finalmente, se imprime esta solución junto con su valor total.

aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
mejor_solucion = poblacion[np.argmax(aptitudes), :]
print("Mejor solución:", mejor_solucion, "Valor:", np.max(aptitudes))

#==============================================================================
# Observaciones del modelo aplicado
# La implementación presentada del algoritmo genético para el problema de la 
# mochila es funcional, pero contiene varios puntos críticos que podrían estar 
# afectando significativamente la calidad de las soluciones generadas. A 
# continuación, se proporciona una revisión académica detallada con sugerencias 
# de mejora y una versión corregida del código, asegurando que el algoritmo 
# encuentre soluciones más cercanas al óptimo y factibles dentro de las 
# restricciones del problema.
#==============================================================================

# I. Diagnóstico y análisis de problemas del código original

# 1. Selección de padres inadecuada. El procedimiento actual de selección de 
# padres usa una forma de selección elitista rígida (selección por mayor aptitud 
# y eliminación directa), lo cual reduce drásticamente la diversidad genética. 
# Esto provoca convergencia prematura y soluciones subóptimas, especialmente si 
# los padres seleccionados no exploran suficientemente el espacio de búsqueda.

# 2. Baja Diversidad en Mutaciones La tasa de mutación de 0.01 es extremadamente 
# baja para una población pequeña y una representación binaria. La mutación 
# limitada dificulta escapar de óptimos locales.

# 3. Cruce Determinista Simple El uso de un único punto de cruce fijo (la mitad 
# del cromosoma) reduce la capacidad de recombinar adecuadamente los genes, 
# limitando la exploración del espacio de soluciones.

# 4. Falta de Verificación de Factibilidad Aunque se penalizan soluciones 
# inviables asignando aptitud 0, no se evita que éstas dominen la población 
# inicial o durante la evolución, degradando la calidad global.
#==============================================================================

# II. Propuesta de mejora al código
# Se introducen las siguientes mejoras:
# Se cambia la selección de padres por torneo binario, promoviendo diversidad.
# Se emplea cruce de un punto aleatorio para mayor variabilidad.
# Se incrementa la tasa de mutación al 0.05 (puede ajustarse dinámicamente).
# Se introduce una verificación explícita de factibilidad en cada individuo.
# Se imprime la mejor solución por generación para trazabilidad.
#==============================================================================

