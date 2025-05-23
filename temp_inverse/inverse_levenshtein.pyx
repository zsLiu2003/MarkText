import numpy as np
from math import log

def levenshtein(x, y, gamma=0.0):
    n = len(x)
    m = len(y.shape[1])  # For 2D arrays, this gets the second dimension size.

    A = np.zeros((n+1, m+1), dtype=np.float32)
    
    for i in range(0, n+1):
        for j in range(0, m+1):
            if i == 0:
                A[i][j] = j * gamma
            elif j == 0:
                A[i][j] = i * gamma
            else:
                cost = log(1 - y[j-1, x[i-1]])
                A[i][j] = A[i-1][j] + gamma
                if A[i][j-1] + gamma < A[i][j]:
                    A[i][j] = A[i][j-1] + gamma
                if A[i-1][j-1] + cost < A[i][j]:
                    A[i][j] = A[i-1][j-1] + cost

    return A[n][m]