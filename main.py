# Created: 10-05-2024 by Sima Moshafi 
# Project 1: QR Factorization with Householder Reflections
# In this project, we will implement the QR factorization of a matrix A using Householder reflections.
# We will then use the resulting Q and R matrices to calculate the backward error of the factorization and compare it to the error in sin(pi).
# My reference for this project is the book "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III.


import numpy as np
from HHQR import house, formQ, inf_norm

# Testing the factorization
j = np.arange(21)
h = 2 / 5
x_j = -4 + j * h

# Form the matrix A for a degree 8 polynomial using the Vandermonde matrix
A = np.column_stack([x_j**i for i in range(9)])  # 9 columns for polynomial of degree 8 

# Do QR factorization using Householder reflections
W, R = house(A)
Q = formQ(W)

# Compute backward error manually
QR_minus_A = Q @ R - A
backward_error = inf_norm(QR_minus_A)

# Compute orthogonality error manually
QTQ_minus_I = Q.T @ Q - np.eye(Q.shape[0])
orthogonality_error = inf_norm(QTQ_minus_I)

print(f"Backward error (manual): {backward_error}")
print(f"Orthogonality error: {orthogonality_error}")


