import numpy as np
import matplotlib.pyplot as plt

from utils import vech_to_mat, mat_to_vec, mat_to_vech, duplication_matrix, spectral_range

pred_grad_f_list = []
pred_grad_ts_f_list = []
cor_grad_f_list = []
s_hat_list = []
theta_hat_list = []
f_list = []
spectral_range_list = []

def predict(s_hat, Theta_hat, Theta_hat_prev, alpha, h, D):
    s = s_hat.copy()
    S_inv = np.linalg.inv(vech_to_mat(s_hat))
    grad_f = D.T @ mat_to_vec(Theta_hat - S_inv)
    pred_grad_f_list.append(grad_f)
    grad_ts_f = D.T @ mat_to_vec(Theta_hat - Theta_hat_prev)
    pred_grad_ts_f_list.append(grad_ts_f)
    s -= 2 * alpha * (grad_f + h * grad_ts_f)
    return project(s)

def correct(s_hat, Theta_hat, beta, D):
    s = s_hat.copy()
    S_inv = np.linalg.inv(vech_to_mat(s_hat))
    grad_f = D.T @ mat_to_vec(Theta_hat - S_inv)
    cor_grad_f_list.append(grad_f)
    s -= beta * grad_f
    return project(s)

def project(s):
    S = vech_to_mat(s)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 0.01)
    S = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return mat_to_vech(S)

def online_graph_learning(data, P, C, alpha, beta, gamma, S_true):
    N = data.shape[1]
    Theta_hat = np.cov(data[:10, :], rowvar=False)
    s_hat = mat_to_vech(np.linalg.inv(Theta_hat))
    Theta_hat_prev = Theta_hat
    theta_hat_list.append(mat_to_vec(Theta_hat))
    nses = []
    D = duplication_matrix(N)

    for t in range(N, data.shape[0]):
        print(t)
        for _ in range(P):
            s_hat = predict(s_hat, Theta_hat, Theta_hat_prev, alpha, 1, D)

        x_t = data[t, :]
        Theta_hat_prev = Theta_hat
        Theta_hat = gamma * Theta_hat_prev + (1 - gamma) * np.outer(x_t, x_t)
        theta_hat_list.append(mat_to_vec(Theta_hat))

        for _ in range(C):
            s_hat = correct(s_hat, Theta_hat, beta, D)

        S_hat = vech_to_mat(s_hat)
        s_hat_list.append(s_hat)
        nse = np.linalg.norm(S_hat - S_true, ord='fro') ** 2 / np.linalg.norm(S_true, ord='fro') ** 2
        nses.append(nse)
        f = -np.log(np.linalg.det(S_hat)) + np.trace(S_hat @ Theta_hat)
        f_list.append(f)

    return vech_to_mat(s_hat), nses, f_list

def generate_data(N, T, seed=24):
    np.random.seed(seed)
    X = np.random.rand(N, N)
    Theta_true = np.cov(X)

    eigvals, eigvecs = np.linalg.eigh(Theta_true)
    Theta_true = eigvecs @ np.diag([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) @ eigvecs.T

    S_true = np.linalg.inv(Theta_true)
    data = np.random.multivariate_normal(mean=np.zeros(N), cov=Theta_true, size=T)

    return data, S_true

if __name__ == "__main__":

    N = 10
    T = 20000

    P = 1
    C = 1

    alpha = 0.001
    beta = 0.001
    gamma = 0.999

    data, S_true = generate_data(N, T, seed=115)

    print(spectral_range(S_true))

    S_hat, nses, f_list = online_graph_learning(data, P, C, alpha, beta, gamma, S_true)

    plt.figure(figsize=(8, 6))
    plt.semilogy(nses)
    plt.xlabel('Iteration')
    plt.ylabel('NSE (log scale)')
    plt.title('Online Graph Learning Convergence')
    plt.grid(True)
    plt.show()

