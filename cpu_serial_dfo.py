# This code is adapted from "COMP1805 - Natural Computing"
# Author: Dr. Mohammad Majid al-Rifaie
# Date : 2022
# Availability : Moodle - DFO
# Serial DFO on CPU

# importing the libraries for array creation, equations and execution time measurement
import numpy as np
import math
import time

# All the Fitness functions used to benchmark the code

# Sphere Function
def f_sphere(x):  # x (vector) - one fly with its dimensionality values
    sum = 0.0
    for i in range(len(x)):
        sum = sum + np.power(x[i], 2)
    return sum

# Rastrigin Function
def f_rast(x): # x (vector) - one fly with its dimensionality values
    sum=0.0
    for i in range(len(x)):
        sum=sum+(np.power(x[i],2)-10*math.cos(2*math.pi*x[i])+10)
    return sum

# Goldstein-Price Function
def f_gold(x): # x (vector) - one fly with its dimensionality values
    x1 = x[0]
    x2 = x[1]
    part1 = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
    part2 = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
    return part1*part2

# Ackley Function
def f_ackley(x): # x (vector) - one fly with its dimensionality values
    square_sum=0.0
    cos_sum=0.0
    sum=0.0
    for i in range(len(x)):
        square_sum+=x[i]**2
        cos_sum+=np.cos(2.0 * np.pi *x[i])
    sum1 = -20.0 * np.exp(-0.2 * np.sqrt(square_sum / len(x)))
    sum2 = -np.exp(cos_sum / len(x))
    sum = sum1 + sum2 + 20.0 + np.exp(1)
    return sum

# The main body of the Code


# Configuring the variables used inside the algorithm
N = 200  # Population Size
D = 30  # Dimensionality
delta = 0.001  # Disturbance Threshold
maxIterations = 1000  # Number of iterations desired
lowerB = [-5.12] * D  # Lower input bound in D dimensions
upperB = [5.12] * D  # Upper input bound in D dimensions

# Phase 1: Initialisation of arrays
X = np.empty([N, D])  # Empty array of size (N,D) for flies
fitness = [None] * N  # Empty array of size N for the fitness of flies

# Phase 2: Populating the flies array with random values between upper and lower bounds
for i in range(N):
    for d in range(D):
        X[i, d] = np.random.uniform(lowerB[d], upperB[d])

# starting point for the timer
start=time.perf_counter()

# Phase 3: Going through each iteration and updating the flies position
# MAIN DFO LOOP
for itr in range(maxIterations):
    # Fitness evaluation for each fly
    for i in range(N):
        fitness[i] = f_sphere(X[i,])                                     # CHANGE FUNCTION NAME HERE
    s = np.argmin(fitness)  # Finding the best fly in this iteration

    # Visual aid - printing the best fly every 100 iterations
    if (itr % 100 == 0 or itr == maxIterations-1):
        print("Iteration:", itr, "\tBest fly index:", s,
              " \tFitness value:", fitness[s])

    # For loop for updating each fly in the population
    for i in range(N):
        if i == s: continue  # Best fly position won't change

        # Identifying best neighbour
        left = (i - 1) % N
        right = (i + 1) % N
        bNeighbour = right if fitness[right] < fitness[left] else left

        # Updating the value for each dimension
        for d in range(D):

            # Local minima stagnation avoidance
            if (np.random.rand() < delta):
                X[i, d] = np.random.uniform(lowerB[d], upperB[d])
                continue;

            u = np.random.rand()
            # Updating the dimension
            X[i, d] = X[bNeighbour, d] + u * (X[s, d] - X[i, d])

            # If the updated value is out of bounds
            if X[i, d] < lowerB[d] or X[i, d] > upperB[d]:
                X[i, d] = np.random.uniform(lowerB[d], upperB[d])

# Phase 4: Final evaluation
for i in range(N): fitness[i] = f_sphere(X[i,])                           # CHANGE FUNCTION NAME HERE
# Final best fly
s = np.argmin(fitness)

# Printing the fitness of the best fly and its position
print("\nFinal best fitness:\t", fitness[s])
print("\nBest fly position:\n", X[s,])

# end point for the timer
finish = time.perf_counter()

# Printing the execution time
print(f'Finished in {round(finish - start, 2)} second(s)')