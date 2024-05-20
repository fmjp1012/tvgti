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

def correct(s_hat, Theta_hat, beta, D, pinv_rcond):
    s = s_hat.copy()
    S_inv = np.linalg.pinv(vech_to_mat(s_hat), pinv_rcond)
    grad_f = D.T @ mat_to_vec(Theta_hat - S_inv)
    s -= beta * grad_f
    return project(s)

def project(s):
    S = vech_to_mat(s)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-16)
    S = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return mat_to_vech(S)

def normalize_covariance(cov_matrix):
    variances = np.diagonal(cov_matrix)
    stddevs = np.sqrt(variances)
    diagonal_inv = np.diag(1 / stddevs)
    corr_matrix = diagonal_inv @ cov_matrix @ diagonal_inv
    return corr_matrix

def compare_projections(A):
    # SDP射影
    X_sdp = sdp(A)
    
    # 固有値射影
    s = mat_to_vech(A)
    X_eig = vech_to_mat(project(s))
    
    # フロベニウスノルムの差を計算
    diff_sdp = np.linalg.norm(A - X_sdp, 'fro')
    diff_eig = np.linalg.norm(A - X_eig, 'fro')
    
    print(f"元の行列とSDP射影行列とのフロベニウスノルムの差: {diff_sdp}")
    print(f"元の行列と固有値射影行列とのフロベニウスノルムの差: {diff_eig}")

def double_half_elements(matrix):
    n = matrix.shape[0]
    count = (n * n) // 2  # 半分の要素数を計算
    indices = np.triu_indices(n)  # 上三角行列のインデックスを取得

    # ランダムにインデックスを選ぶ
    selected_indices = np.random.choice(len(indices[0]), count, replace=False)
    
    for idx in selected_indices:
        i, j = indices[0][idx], indices[1][idx]
        matrix[i, j] *= 2
        if i != j:  # 対称な位置の要素も選んで2倍にする
            matrix[j, i] *= 2

    return matrix