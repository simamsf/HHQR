# Created: 10-05-2024 by Sima Moshafi 
# Project 1: QR Factorization with Householder Reflections
# In this project, we will implement the QR factorization of a matrix A using Householder reflections.
# We will then use the resulting Q and R matrices to calculate the backward error of the factorization and compare it to the error in sin(pi).
# My reference for this project is the book "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III.


import numpy as np
import matplotlib.pyplot as plt
from HHQR import house, formQ, inf_norm, implicit_QT_b, back_substitution, evaluate_polynomial

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


# Task 3: Solve the least squares problem using the QR factorization
# Create vector b for sin(x_j)
b = np.sin(x_j)

# Compute Q^T * b
Q_T_b = implicit_QT_b(W, b.copy())  # Compute Q^T * b using the Householder vectors in W

# Solve for c
c = back_substitution(R, Q_T_b)
print("Solution vector c:", c)


# Task 4: Evaluate the polynomial approximation at pi and plot the results
# Compute p(pi)
pi_approx = evaluate_polynomial(c, np.pi)

# Calculate the error |p(pi) - sin(pi)|
error = abs(pi_approx - 0)  # Since sin(pi) is approximately 0
print(f"Error in approximating sin(pi): {error}")

# Plot p(x) and sin(x) on the interval [-4, 4]
x_vals = np.linspace(-4, 4, 400)
p_vals = [evaluate_polynomial(c, x) for x in x_vals]
sin_vals = np.sin(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, p_vals, color="teal", label="$p(x)$", linestyle="--", linewidth=2)
plt.plot(x_vals, sin_vals, color="orange", label="$\sin(x)$", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Approximation of $\sin(x)$")
plt.legend()
plt.grid(True)
plt.show()


