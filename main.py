# Created: 10-05-2024 by Sima Moshafi 
# Project 1: QR Factorization with Householder Reflections
# In this project, we will implement the QR factorization of a matrix A using Householder reflections.
# We will then use the resulting Q and R matrices to calculate the backward error of the factorization and compare it to the error in sin(pi).
# My reference for this project is the book "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III.

import numpy as np

from numpy import typing as npt

MyArray = npt.NDArray[np.float64]
def house(A: MyArray) -> tuple[MyArray, MyArray]:
    '''
    Gets the QR factorization of matrix A using Householder reflections.
    
    Outputs: W and R where W.
    '''
    
    m, n = A.shape # A.shape returns a tuple containing the number of rows and columns in A
    W = np.zeros_like(A) # matrix W with the same shape as A but I fill it with zeros.
    R = A.copy() # creates an independent copy of A and stores it in R.
    # This is where where I will perform transformations and reduce it to a triangular form.
    
    for k in range(n): # k starts at 0 and goes up to n-1, where n is the number of columns of A.
        x = R[k:, k] # vector x: the part of column k of R starting from row k down to the end.
        
        # Compute the Householder vector
        e1 = np.zeros_like(x) # The vector e1 should represent (1, 0, 0, …) but initialized as a zero
        # we can easily set its first element to the norm x value with the right sign.
        e1[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1) # changes the first element to the norm with the
        # sign of the first element of x. If x[0] is negative, the norm is multiplied by -1. Otherwise by 1.
        v_k = e1 + x # This adds the vector x to e1 to construct the Householder vector v_k.
        v_k = v_k / np.linalg.norm(v_k) # Normalizes the Householder vector v_k.
        
        # Apply the reflection to R
        R[k:, k:] -= 2 * np.outer(v_k, v_k @ R[k:, k:]) # applies the Householder transformation to the
        # lower right part of R.
        # I am calling it R to be stored and be different from the original matrix A. But it is the
        # subsequent matrix of A after applying the Householder transformation.
        
        # Store the Householder vector
        W[k:, k] = v_k # stores all the Householder vectors in the matrix W.
    return W, R



def implicit_QT_b(W: MyArray, b: MyArray) -> MyArray:
    # Q is the orthogonal matrix constructed from the Householders stored
    # in matrix W (instead of explicitly forming the matrix). b is a vector.
    '''Implicit calculation of Q^*b using v_k, the Householder vectors from matrix W.'''
    m, n = W.shape 
    for k in range(n): # like saying columns from k=1 to n of matrix W in the book.
        v_k = W[k:, k] # reuse the Householder vectors stored in W.
        b[k:] -= 2 * np.outer(v_k, v_k).dot(b[k:]) # apply the Householder transformation to
        # b as b_k:m = b_k:m - 2v_k(v_k^* b_k:m)
    return b



def implicit_Q_x(W: MyArray, x: MyArray) -> MyArray: 
    """Implicit calculation of Qx using Householder vectors from matrix W."""
    m, n = W.shape
    for k in range(n-1, -1, -1): # starting from the last column of W (k=n-1) and going
        # backwards down to the first column (k=0). This is starting from the last Householder
        # vector and working toward the first one.
        v_k = W[k:, k]
        x[k:] -= 2 * np.outer(v_k, v_k).dot(x[k:]) # x_k:m = x_k:m - 2v_k(v_k^* x_k:m)
    return x



def formQ(W: MyArray) -> MyArray:
    '''
    Takes W from function house(A) and generates othogonal matrix Q_m*m
    '''
    m, n = W.shape
    Q = np.eye(m) # will update by applying the Householder reflections.
    for k in range(n-1, -1, -1): # Householder vectors from k=n-1 down to k=0.
        v_k = W[k:, k] # separates colomn k of W from row k down to the end.
        # Q_k:m = Q_k:m - 2v_k(v_k^* Q_k:m) is the applied Householder transformation.
        Q[k:, :] -= 2 * np.outer(v_k, np.dot(v_k, Q[k:, :]))
    return Q # outputs the orthogonal matrix Q formed from the Householder vectors stored in W.



def inf_norm(A: MyArray) -> float:
    """Computes the infinity norm of matrix A manually."""
    return max(np.sum(np.abs(A), axis=1))
    # return max(sum(abs(A[i, j]) for j in range(A.shape[1])) for i in range(A.shape[0]))



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


