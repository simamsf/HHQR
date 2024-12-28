import numpy as np

from numpy import typing as npt

MyArray = npt.NDArray[np.float64]


def house(A: MyArray) -> tuple[MyArray, MyArray]:
    '''
    Gets the QR factorization of matrix A using Householder reflections.
    Outputs: W and R where W.
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


def implicit_QT_b(W: MyArray, b: MyArray) -> MyArray:
    '''Implicit calculation of Q^*b using v_k, the Householder vectors from matrix W.'''
    m, n = W.shape
    for k in range(n):
        v_k = W[k:, k]
        b[k:] -= 2 * np.outer(v_k, v_k).dot(b[k:])
    return b


def implicit_Q_x(W: MyArray, x: MyArray) -> MyArray:
    """Implicit calculation of Qx using Householder vectors from matrix W."""
    m, n = W.shape
    for k in range(n-1, -1, -1):
        v_k = W[k:, k]
        x[k:] -= 2 * np.outer(v_k, v_k).dot(x[k:])
    return x


def formQ(W: MyArray) -> MyArray:
    '''
    Takes W from function house(A) and generates othogonal matrix Q_m*m
    '''
    m, n = W.shape
    Q = np.eye(m)
    for k in range(n-1, -1, -1):
        v_k = W[k:, k]
        Q[k:, :] -= 2 * np.outer(v_k, np.dot(v_k, Q[k:, :]))
    return Q


def inf_norm(A: MyArray) -> float:
    """Computes the infinity norm of matrix A manually."""
    return max(np.sum(np.abs(A), axis=1))


# Solve Rc = Q^T * b using back-substitution
def back_substitution(R, y):
    """Solve the equation Rx = y for x, where R is an upper triangular matrix."""
    n = R.shape[1]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x


# Evaluate p(x) at a given value using Horner's method
def evaluate_polynomial(coeffs, x):
    result = 0
    for c in reversed(coeffs):
        result = result * x + c
    return result
