# Created: 10-05-2024 by Sima Moshafi 
# Project 1: QR Factorization with Householder Reflections.
# We will then use the QR to calculate the backward error of the factorization and compare it to the error in sin(pi).
# My reference for this project is the book "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III.

import numpy as np

def house(A):
    '''
    Gets the QR factorization of matrix A using Householder reflections.
    
    Outputs: W and R where W is the matrix of the succesive Householder vectors v_k and R is the triangular matrix.
    '''
    m, n = A.shape
    W = np.zeros_like(A)
    R = A.copy()
    
    for k in range(n):
        x = R[k:, k]
        
        # Compute the Householder vector
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1) 
        v_k = e1 + x 
        v_k = v_k / np.linalg.norm(v_k)
        
        # Apply the reflection to R
        R[k:, k:] -= 2 * np.outer(v_k, v_k @ R[k:, k:])
        
        # Store the Householder vector
        W[k:, k] = v_k
    
    return W, R



def implicit_QT_b(W, b):
    '''Implicit calculation of Q^*b using v_k, the Householder vectors from matrix W.'''
    m, n = W.shape 
    for k in range(n):
        v_k = W[k:, k] 
        b[k:] -= 2 * np.outer(v_k, v_k).dot(b[k:]) 
    return b



def implicit_Q_x(W, x): 
    """Implicit calculation of Qx using Householder vectors from matrix W."""
    m, n = W.shape
    for k in range(n-1, -1, -1):
        v_k = W[k:, k]
        x[k:] -= 2 * np.outer(v_k, v_k).dot(x[k:])
    return x



def formQ(W):
    '''
    Takes W from function house(A) and generates othogonal matrix Q_m*m
    '''
    m, n = W.shape
    Q = np.eye(m)
    for k in range(n-1, -1, -1):
        v_k = W[k:, k] 
        Q[k:, :] -= 2 * np.outer(v_k, np.dot(v_k, Q[k:, :]))
    return Q