# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:41:02 2023

@author: Jaime
"""
#==============================================================================
# DETERMINAR VARIABLES IMPORTANTES USANDO ALGORITMOS GENÉTICOS

# Este algoritmo emplea un enfoque basado en algoritmos genéticos para 
# identificar las características más importantes de un conjunto de datos en 
# un problema de clasificación. El proceso comienza cargando y preprocesando 
# un conjunto de datos relacionado con la potabilidad del agua, donde se 
# llenan los valores faltantes y se estandarizan las características. Luego, 
# se divide el conjunto de datos en conjuntos de entrenamiento y prueba. Se 
# utiliza un algoritmo genético para seleccionar subconjuntos óptimos de 
# características, donde cada individuo en la población representa una 
# combinación de características (con un bit de 1 indicando la inclusión de 
# la característica y un bit de 0 indicando su exclusión). La aptitud de cada 
# individuo se evalúa entrenando un clasificador RandomForest con las 
# características seleccionadas y midiendo su precisión en el conjunto de 
# prueba. A lo largo de 10 generaciones, se seleccionan, cruzan y mutan 
# individuos para mejorar la aptitud, con el objetivo de maximizar la 
# precisión del modelo. Al final, el algoritmo identifica y presenta la mejor 
# combinación de características seleccionadas, junto con su valor de aptitud, 
# lo que indica qué subconjunto de variables contribuye más efectivamente al 
# rendimiento del clasificador.
#==============================================================================
!pip install deap

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from deap import creator, base, tools, algorithms
import random

# Set random seed for reproducibility
np.random.seed(0)

# Load the dataset
data = pd.read_csv('water_potability.csv')

# Fill missing values with median
data.fillna(data.median(), inplace=True)

# Split the dataset into features (X) and target (y)
X = data.drop(columns=['Potability'])
y = data['Potability']

# Standardize the features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Number of features
n_features = X_train.shape[1]

# Create classes for fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create a toolbox
toolbox = base.Toolbox()

# Attribute generator 
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the fitness function
def evalOneMax(individual):
    # Transform the individual into a boolean mask
    mask = np.array(individual, dtype=bool)
    
    # If all elements are False, return a very low fitness
    if not np.any(mask):
        return -np.inf,
    
    # Train and evaluate a RandomForestClassifier using only the selected features
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train.iloc[:, mask], y_train)
    y_pred = clf.predict(X_test.iloc[:, mask])
    return accuracy_score(y_test, y_pred),

# Register the fitness function in the toolbox
toolbox.register("evaluate", evalOneMax)

# Register genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create an initial population
pop = toolbox.population(n=50)

# Define parameters for the genetic algorithm
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# Run the genetic algorithm
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, 
                               stats=stats, halloffame=hof, verbose=True)

# Get the best individual
best_ind = hof[0]
print("Best individual is ", best_ind, " with fitness: ", best_ind.fitness.values)
