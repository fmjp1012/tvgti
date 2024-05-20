import numpy as np

from cvx.sdp import *

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

def predict(s_hat, Theta_hat, Theta_hat_prev, alpha, h, D, pinv_rcond):
    s = s_hat.copy()
    S_inv = np.linalg.pinv(vech_to_mat(s_hat), pinv_rcond)
    grad_f = D.T @ mat_to_vec(Theta_hat - S_inv)
    grad_ts_f = D.T @ mat_to_vec(Theta_hat - Theta_hat_prev)
    s -= 2 * alpha * (grad_f + h * grad_ts_f)
    return project(s)

def correct(s_hat, Theta_hat, beta, D):
    s = s_hat.copy()
    S_inv = np.linalg.inv(vech_to_mat(s_hat))
    grad_f = D.T @ mat_to_vec(Theta_hat - S_inv)
    s -= beta * grad_f
    return project(s)

def project(s):
    S = vech_to_mat(s)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-2)
    S = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return mat_to_vech(S)

def spectral_range(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_range = np.max(eigenvalues) - np.min(eigenvalues)
    return spectral_range