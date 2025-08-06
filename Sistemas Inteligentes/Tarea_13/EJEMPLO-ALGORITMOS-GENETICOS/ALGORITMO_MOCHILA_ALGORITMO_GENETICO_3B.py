
#==============================================================================
# Código 4 de la mochila con multi-eleccción

# El código implementa un algoritmo genético para resolver el problema de la 
# mochila con repetición de ítems, donde se busca seleccionar una combinación 
# de artículos (cada uno con un peso y un valor) que maximice el valor total 
# sin superar una capacidad máxima. A diferencia de la mochila 0-1, aquí se 
# permite seleccionar múltiples copias de un mismo ítem, representando cada 
# individuo como un vector de enteros no negativos. El proceso comienza con la 
# generación de una población aleatoria de soluciones factibles, asegurada 
# mediante una función de reparación que reduce iterativamente los ítems menos 
# eficientes (menor valor/peso) hasta cumplir la restricción de capacidad. Cada 
# generación incluye el cálculo de la aptitud (valor total si el peso no excede 
# la capacidad), selección de padres mediante torneo binario, reproducción a 
# través de cruce uniforme y mutación basada en incremento o decremento de las 
# cantidades de ítems. Tanto el cruce como la mutación aplican también la 
# reparación para garantizar que la descendencia sea válida. Tras un número 
# determinado de generaciones, el algoritmo imprime la mejor solución encontrada, 
# incluyendo su valor total y el peso asociado. Este enfoque busca optimizar de 
# manera evolutiva una función objetivo bajo restricciones duras, manteniendo 
# la factibilidad de las soluciones en todo momento.
#==============================================================================

# Cargar librerías
import numpy as np

#---------------------------------------
# Datos del problema
# En esta sección inicial se definen los elementos fundamentales del problema 
# de la mochila. Se incluyen dos vectores: uno con los pesos de los artículos 
# disponibles y otro con sus respectivos valores. También se establece la 
# capacidad máxima de la mochila, que representa la restricción principal del 
# problema. Además, se configuran los parámetros del algoritmo genético: el 
# tamaño de la población, la tasa de cruce (porcentaje de la población que se 
# reproduce), la tasa de mutación (probabilidad de alteración genética por gen), 
# el número total de generaciones del proceso evolutivo y una estimación del 
# número máximo de veces que un ítem puede repetirse, calculada con base en el 
# peso más pequeño y la capacidad disponible.
#---------------------------------------
pesos = np.array([12, 2, 1, 4, 1, 5, 3, 2, 1, 4])
valores = np.array([4, 2, 1, 10, 2, 8, 5, 2, 1, 1])
capacidad = 25

tamaño_poblacion = 100
tasa_cruce = 0.8
tasa_mutacion = 0.1
num_generaciones = 100

# Máximo número posible de repeticiones para cada ítem
max_repeticiones = capacidad // np.min(pesos)

#---------------------------------------
# Función de reparación de individuos
# Dado que cada individuo en este problema puede representar una solución no 
# factible (es decir, que exceda el límite de peso), se define una función de 
# reparación que corrige este tipo de situaciones. Esta función evalúa el peso 
# total de un individuo y, si supera la capacidad, procede a disminuir la 
# cantidad de los ítems que presentan menor eficiencia (definida como el 
# cociente valor/peso). El proceso continúa iterativamente hasta que el 
# individuo se ajusta a la restricción de capacidad. De esta manera, se asegura 
# que todas las soluciones en el proceso evolutivo cumplan con los requisitos 
# del problema y puedan ser evaluadas correctamente por la función de aptitud.
#---------------------------------------
def reparar_individuo(ind):
    while np.sum(ind * pesos) > capacidad:
        # Reducción del ítem menos eficiente
        eficiencia = valores / pesos
        idx_menos_eficiente = np.argmin(eficiencia + (ind == 0) * 1e6)
        if ind[idx_menos_eficiente] > 0:
            ind[idx_menos_eficiente] -= 1
        else:
            break
    return ind

#---------------------------------------
# Función de aptitud
# La función de aptitud es el núcleo del proceso de evaluación en un algoritmo 
# genético. Para cada individuo, se calcula el valor total de los ítems 
# seleccionados y el peso correspondiente. Si el peso total se encuentra dentro 
# del límite permitido, la función retorna el valor como medida de calidad 
# (aptitud) de la solución. En caso contrario, si el peso excede la capacidad, 
# la función retorna 0, lo que penaliza a los individuos no válidos. Esta 
# función actúa como criterio de selección natural, favoreciendo a las soluciones 
# más rentables que cumplan con la restricción de peso.
#---------------------------------------
def aptitud(individuo):
    peso_total = np.sum(individuo * pesos)
    valor_total = np.sum(individuo * valores)
    return valor_total if peso_total <= capacidad else 0

#---------------------------------------
# Selección por torneo
# En esta etapa se seleccionan los padres que participarán en el proceso de 
# reproducción. Se emplea un método de selección por torneo binario, en el cual 
# se eligen dos individuos al azar de la población y se selecciona al de mayor 
# aptitud como ganador. Este procedimiento se repite hasta obtener el número 
# deseado de padres, determinado como un porcentaje del tamaño total de la 
# población. La selección por torneo favorece a los individuos más aptos, pero 
# introduce aleatoriedad suficiente para mantener diversidad genética dentro 
# del proceso evolutivo.
#---------------------------------------
def seleccionar_padres(poblacion, aptitudes, num_padres):
    padres = []
    for _ in range(num_padres):
        i, j = np.random.choice(range(len(poblacion)), size=2, replace=False)
        ganador = poblacion[i] if aptitudes[i] > aptitudes[j] else poblacion[j]
        padres.append(ganador.copy())
    return np.array(padres)

#---------------------------------------
# Cruce uniforme + reparación
# Esta función genera nuevos individuos (descendientes) mediante la combinación 
# genética de dos padres seleccionados previamente. El cruce se realiza de 
# manera uniforme: para cada gen, se elige aleatoriamente si se toma del primer 
# padre o del segundo, generando un hijo híbrido que hereda características de 
# ambos progenitores. Como este proceso puede generar soluciones no válidas 
# (es decir, que excedan la capacidad), cada descendiente es sometido a la 
# función de reparación inmediatamente después del cruce. Este procedimiento 
# garantiza que las nuevas soluciones generadas sean factibles y puedan ser 
# evaluadas sin penalización.
#---------------------------------------
def cruce(padres, tamaño_poblacion):
    num_descendencia = tamaño_poblacion - len(padres)
    descendencia = np.empty((num_descendencia, padres.shape[1]), dtype=int)
    for k in range(num_descendencia):
        padre1 = padres[np.random.randint(len(padres))]
        padre2 = padres[np.random.randint(len(padres))]
        mascara = np.random.randint(2, size=padres.shape[1])
        hijo = np.where(mascara == 1, padre1, padre2)
        hijo = reparar_individuo(hijo)
        descendencia[k] = hijo
    return descendencia

#---------------------------------------
# Mutación por incremento/decremento + reparación
# La mutación introduce pequeñas alteraciones aleatorias en los individuos 
# descendientes para mantener la variabilidad genética de la población y evitar 
# la convergencia prematura. En este caso, la mutación consiste en incrementar 
# o decrementar la cantidad de veces que se selecciona un ítem, con una cierta 
# probabilidad por cada gen. Tras aplicar estos cambios, se verifica nuevamente 
# si el individuo modificado respeta la capacidad máxima. En caso contrario, 
# se repara usando la función previamente definida. Así, esta fase permite 
# explorar nuevas combinaciones de soluciones sin violar las restricciones del 
# problema.
#---------------------------------------
def mutacion(descendencia):
    for idx in range(descendencia.shape[0]):
        for gen in range(descendencia.shape[1]):
            if np.random.rand() < tasa_mutacion:
                cambio = np.random.choice([-1, 1])
                descendencia[idx, gen] = max(0, descendencia[idx, gen] + cambio)
        descendencia[idx] = reparar_individuo(descendencia[idx])
    return descendencia

#---------------------------------------
# Inicialización controlada con reparación
# La población inicial del algoritmo se genera mediante la creación aleatoria 
# de individuos, donde cada gen (ítem) toma un número entero aleatorio entre 0 
# y el máximo número de repeticiones estimado. Luego, cada individuo es 
# reparado para asegurar que cumpla la restricción de capacidad desde el 
# principio. Esta inicialización controlada evita que el algoritmo comience con 
# soluciones no válidas que puedan obstaculizar el proceso evolutivo, y 
# proporciona una base diversa de soluciones iniciales desde la cual evolucionar 
# hacia mejores combinaciones.
#---------------------------------------
poblacion = []
for _ in range(tamaño_poblacion):
    individuo = np.random.randint(0, max_repeticiones + 1, size=len(pesos))
    individuo = reparar_individuo(individuo)
    poblacion.append(individuo)
poblacion = np.array(poblacion)

#---------------------------------------
# Algoritmo genético
# Esta sección constituye el ciclo evolutivo del algoritmo, que se repite 
# durante un número predefinido de generaciones. En cada iteración se calcula 
# la aptitud de todos los individuos, se seleccionan los padres mediante torneo, 
# se genera la descendencia por cruce uniforme, se aplica la mutación con 
# reparación y se conforma la nueva población reemplazando los antiguos 
# individuos con los nuevos. También se imprime el mejor individuo de cada 
# generación para monitorear el progreso del algoritmo. Este proceso de 
# selección, reproducción, variación y reemplazo emula el mecanismo evolutivo de 
# la naturaleza, guiado por el principio de supervivencia del más apto.
#---------------------------------------
for generacion in range(num_generaciones):
    aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
    padres = seleccionar_padres(poblacion, aptitudes, int(tasa_cruce * tamaño_poblacion))
    descendencia = cruce(padres, tamaño_poblacion)
    descendencia = mutacion(descendencia)
    poblacion[:len(padres)] = padres
    poblacion[len(padres):] = descendencia

    mejor = poblacion[np.argmax(aptitudes)]
    print(f"Generación {generacion + 1}: Valor = {np.max(aptitudes)}, Solución = {mejor}")

#---------------------------------------
# Resultado final
# Una vez completadas todas las generaciones, el algoritmo evalúa la población 
# final para determinar cuál es el mejor individuo encontrado. Se imprime su 
# configuración (es decir, cuántas veces se eligió cada ítem), el valor total 
# alcanzado y el peso correspondiente. Esta salida representa la solución óptima 
# (o cuasi óptima) obtenida por el algoritmo genético dentro del espacio de 
# búsqueda definido y bajo las restricciones del problema de la mochila con 
# repetición.
#---------------------------------------
aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
mejor_solucion = poblacion[np.argmax(aptitudes)]
print("=============================================")
print("Mejor solución final:", mejor_solucion)
print("Valor total:", np.max(aptitudes))
print("Peso total:", np.sum(mejor_solucion * pesos))
