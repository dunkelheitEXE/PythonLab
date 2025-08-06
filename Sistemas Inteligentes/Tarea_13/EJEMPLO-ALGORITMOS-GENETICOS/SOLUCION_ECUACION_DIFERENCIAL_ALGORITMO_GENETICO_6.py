# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:07:21 2023

@author: Jaime
"""
#==============================================================================
# SOLUCIÓN DE UNA ECUACIÓN DIFERENCIAL USANDO ALGORITMOS GENÉTICOS

# Este algoritmo emplea un enfoque basado en algoritmos genéticos para 
# aproximar la solución de una ecuación diferencial ordinaria (EDO) a través de 
# la búsqueda de un polinomio que mejor se ajuste a la solución de la ecuación. 
# La EDO considerada es simple, dy/dt=y, y el objetivo es encontrar un conjunto 
# de coeficientes polinomiales que minimicen el error entre la derivada 
# numérica del polinomio y la derivada real proporcionada por la ecuación 
# diferencial.

# El proceso comienza definiendo la función de aptitud (fitness), que mide el 
# error acumulado entre la derivada aproximada del polinomio y la derivada real 
# en cada paso temporal. Se genera una población inicial de polinomios con 
# coeficientes aleatorios, y a lo largo de 1000 generaciones, los individuos 
# son evaluados y seleccionados en función de su aptitud. Se aplican operadores 
# genéticos de cruce (combinando partes de dos polinomios) y mutación 
# (alterando aleatoriamente un coeficiente) para explorar nuevas soluciones. Al 
# final, el polinomio con los coeficientes que minimizan el error se considera 
# la mejor aproximación de la solución de la ecuación diferencial. El algoritmo 
# imprime estos coeficientes al finalizar, proporcionando una representación 
# polinómica de la solución.
#==============================================================================
import numpy as np

# Define the differential equation
def dydt(y, t):
    return y

# Define the polynomial function
def poly(t, coeffs):
    return sum([c*t**i for i, c in enumerate(coeffs)])

# Define the fitness function
def fitness(individual, dydt, t0, y0):
    dt = 0.01
    t = t0
    y = y0
    error = 0.0
    for _ in range(100):  # simulate for 100 steps
        y_next = poly(t, individual)
        dy = (y_next - y) / dt
        error += abs(dy - dydt(y, t))
        y = y_next
        t += dt
    return -error

# Define the crossover function
def crossover(individual1, individual2):
    crossover_point = np.random.randint(len(individual1))
    return np.concatenate((individual1[:crossover_point], individual2[crossover_point:]))

# Define the mutation function
def mutate(individual):
    mutation_point = np.random.randint(len(individual))
    individual[mutation_point] += np.random.normal()
    return individual

# Define the parameters of the genetic algorithm
population_size = 100
num_generations = 1000
mutation_rate = 0.01

# Initialize the population randomly
population = [np.random.rand(10) for _ in range(population_size)]  # Polynomials up to degree 9

for _ in range(num_generations):
    # Evaluate the fitness of the population
    fitnesses = [fitness(individual, dydt, 0, 1) for individual in population]

    # Select the best individuals to form the next generation
    next_population = [population[i] for i in np.argsort(fitnesses)[-population_size//2:]]

    # Crossover
    for _ in range(population_size // 2):
        individual1, individual2 = np.random.choice(next_population, 2, replace=False)
        offspring = crossover(individual1, individual2)
        next_population.append(offspring)

    # Mutation
    for i in range(len(next_population)):
        if np.random.rand() < mutation_rate:
            next_population[i] = mutate(next_population[i])

    population = next_population

# Print the coefficients of the best fitting polynomial
print(population[np.argmax(fitnesses)])
