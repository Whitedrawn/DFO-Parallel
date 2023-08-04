# This code is adapted from "COMP1805 - Natural Computing"
# Author: Dr. Mohammad Majid al-Rifaie
# Date : 2022
# Availability : Moodle - DFO
# Partial Parallel on GPU

# importing the libraries for array creation, equations, execution time measurement and cuda kernels
import numpy as np
from numba import cuda
import time
import math

# All the Fitness functions used to benchmark the code

@cuda.jit # function decorator
# Sphere Function KERNEL
def fitness_kernel_sphere(X, fitness):
    # Calculating global thread Id
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # Sum of all dimensionality values
    if i < X.shape[0]:
        sum = 0.0
        for j in range(X.shape[1]):
            sum += X[i, j] ** 2
        fitness[i] = sum


@cuda.jit # function decorator
# Rastrigin Function KERNEL
def fitness_kernel_rastrigin(X, fitness):
    # Calculating global thread Id
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # Sum of all dimensionality values
    if i < X.shape[0]:
        sum = 0.0
        for j in range(X.shape[1]):
            sum += ((X[i, j] ** 2) - 10 * math.cos(2 * math.pi * X[i, j]) + 10)
        fitness[i] = sum


@cuda.jit # function decorator
# Goldstein-Price Function
def fitness_kernel_goldstein(X, fitness):
    # Calculating global thread Id
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # Sum of all dimensionality values
    if i < X.shape[0]:
        x1=X[i,0]
        x2=X[i,1]
        part1=1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
        part2= 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
        fitness[i] = part1*part2


@cuda.jit # function decorator
# Ackley Function KERNEL
def fitness_kernel_ackley(X, fitness):
    # Calculating global thread Id
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # Sum of all dimensionality values
    if i < X.shape[0]:
        square_sum = 0.0
        cos_sum = 0.0
        sum = 0.0
        for j in range(X.shape[1]):
            square_sum += X[i,j] ** 2
            cos_sum += math.cos(2.0 * math.pi * X[i,j])
        sum1 = -20.0 * math.exp(-0.2 * math.sqrt(square_sum / X.shape[1]))
        sum2 = -math.exp(cos_sum /X.shape[1])
        sum=sum1 + sum2 + 20.0 + math.exp(1)
        fitness[i] = sum

# The main body of the Code


# Configuring the variables used inside the algorithm
N = 200  # Population Size
D = 30 # Dimensionality
delta = 0.001  # Disturbance Threshold
maxIterations = 1000  # Number of iterations desired
lowerB = [-5.12] * D  # Lower input bound in D dimensions
upperB = [5.12] * D  # Upper input bound in D dimensions

# GPU configuration for Kernel launch
blockSize = 32; # The usual warp size in modern GPUs
gridSize = 136; # SM dependant (usually 2/4 warps per SM)

# Phase 1: Initialisation of arrays
X = np.empty([N, D])  # Empty array of size (N,D) for flies
fitness = np.empty(N)  # Empty array of size N for the fitness of flies

# Phase 2: Populating the flies array with random values between upper and lower bounds
for i in range(N):
    for d in range(D):
        X[i, d] = np.random.uniform(lowerB[d], upperB[d])

# starting point for the timer
start = time.perf_counter()

# Phase 3: Going through each iteration and updating the flies position
# MAIN DFO LOOP
for itr in range(maxIterations):
    # Manual memory transfer to device
    # Creating array on device and copying the ones from host on them
    X_d = cuda.to_device(X)
    fitness_d = cuda.to_device(fitness)

    # Evaluate fitness in parallel using CUDA kernels
    # Kernel launch
    fitness_kernel_sphere[gridSize, blockSize](X_d, fitness_d)             # CHANGE FUNCTION NAME HERE

    # Manual memory transfer back to host
    fitness = fitness_d.copy_to_host()
    s = np.argmin(fitness)  # Finding the best fly in this iteration

    # Visual aid - printing the best fly every 100 iterations
    if (itr % 100 == 0 or itr == maxIterations - 1):
        print("Iteration:", itr, "\tBest fly index:", s,
              "\tFitness value:", fitness[s])

    # For loop for updating each fly in the population
    for i in range(N):
        if i == s:
            continue  # Best fly position won't change

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
# Evaluate fitness in parallel using CUDA

# Manual memory transfer to device
X_d = cuda.to_device(X)
fitness_d = cuda.to_device(fitness)

# Kernel launch
fitness_kernel_sphere[gridSize, blockSize](X_d, fitness_d)                 # CHANGE FUNCTION NAME HERE

# Manual memory transfer back to host
fitness = fitness_d.copy_to_host()

# Final best fly
s = np.argmin(fitness)

# Printing the fitness of the best fly and its position
print("\nFinal best fitness:\t", fitness[s])
print("\nBest fly position:\n", X[s,])
# end point for the timer
finish = time.perf_counter()

# Printing the execution time
print(f'Finished in {round(finish - start, 2)} second(s)')