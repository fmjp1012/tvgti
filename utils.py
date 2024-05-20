import numpy as np

def mat_to_vec(S):
    return S.flatten()

def mat_to_vech(S):
    return S[np.triu_indices(S.shape[0])]

def vech_to_mat(s):
    N = int(np.sqrt(2 * s.shape[0]))
    S = np.zeros((N, N))
    S[np.triu_indices(N)] = s
    S = S + S.T - np.diag(np.diag(S))
    return S

def duplication_matrix(n):
    vech_indices = []
    idx = 0
    for i in range(n):
        for j in range(i, n):
            vech_indices.append((i, j, idx))
            idx += 1

    D = np.zeros((n*n, len(vech_indices)))
    for i, j, idx in vech_indices:
        D[i*n + j, idx] = 1
        if i != j:
            D[j*n + i, idx] = 1
    return D

def spectral_range(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_range = np.max(eigenvalues) - np.min(eigenvalues)
    return spectral_range

def generate_random_matrix(N, spectral_range):
    A = np.random.rand(N, N)
    A = (A + A.T) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(A @ A.T)
    
    eigenvalues = (eigenvalues - np.min(eigenvalues)) / (np.max(eigenvalues) - np.min(eigenvalues))
    eigenvalues = eigenvalues * spectral_range
    
    matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    return matrix