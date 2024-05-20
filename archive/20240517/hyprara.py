import numpy as np
import matplotlib.pyplot as plt
import optuna

pred_grad_f_list = []
pred_grad_ts_f_list = []
cor_grad_f_list = []
s_hat_list = []
theta_hat_list = []

def predict(s_hat, Theta_hat, Theta_hat_prev, alpha, h, D, pinv_rcond, eigval_threshold):
    s = s_hat.copy()
    S_inv = np.linalg.pinv(vech_to_mat(s_hat), pinv_rcond)
    grad_f = D.T @ mat_to_vec(Theta_hat - S_inv)
    pred_grad_f_list.append(grad_f)
    grad_ts_f = D.T @ mat_to_vec(Theta_hat - Theta_hat_prev)
    pred_grad_ts_f_list.append(grad_ts_f)
    s -= 2 * alpha * (grad_f + h * grad_ts_f)
    return project(s, eigval_threshold)

def correct(s_hat, Theta_hat, beta, D, pinv_rcond, eigval_threshold):
    s = s_hat.copy()
    S_inv = np.linalg.pinv(vech_to_mat(s_hat), pinv_rcond)
    grad_f = D.T @ mat_to_vec(Theta_hat - S_inv)
    cor_grad_f_list.append(grad_f)
    s -= beta * grad_f
    return project(s, eigval_threshold)

def project(s, eigval_threshold):
    S = vech_to_mat(s)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, eigval_threshold)
    S = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return mat_to_vech(S)

def online_graph_learning(data, P, C, alpha, beta, gamma, pinv_rcond, eigval_threshold, S_true):
    N = data.shape[1]
    s_hat = mat_to_vech(np.random.rand(N, N))
    Theta_hat = np.linalg.pinv(S_true, pinv_rcond)
    Theta_hat_prev = Theta_hat
    theta_hat_list.append(mat_to_vec(Theta_hat))
    nses = []
    f_list = []
    D = duplication_matrix(N)

    for t in range(N, data.shape[0]):

        for _ in range(P):
            s_hat = predict(s_hat, Theta_hat, Theta_hat_prev, alpha, 1, D, pinv_rcond, eigval_threshold)

        x_t = data[t, :]
        Theta_hat_prev = Theta_hat
        Theta_hat = gamma * Theta_hat_prev + (1 - gamma) * np.outer(x_t, x_t)
        theta_hat_list.append(mat_to_vec(Theta_hat))

        for _ in range(C):
            s_hat = correct(s_hat, Theta_hat, beta, D, pinv_rcond, eigval_threshold)

        S_hat = vech_to_mat(s_hat)
        s_hat_list.append(s_hat)
        nse = np.linalg.norm(S_hat - S_true, ord='fro') ** 2 / np.linalg.norm(S_true, ord='fro') ** 2
        nses.append(nse)
        f = -np.log(np.linalg.det(S_hat) + 1e-6) + np.trace(S_hat @ Theta_hat)
        f_list.append(f)

    return vech_to_mat(s_hat), nses, f_list

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

def generate_data(N, T, pinv_rcond, seed=2):
    np.random.seed(seed)
    X = np.random.rand(N, N)
    mask = np.random.rand(N, N) < 0.2
    X = X * mask
    Theta_true = normalize_covariance(X @ X.T)

    S_true = np.linalg.pinv(Theta_true, pinv_rcond)
    data = np.random.multivariate_normal(mean=np.zeros(N), cov=Theta_true, size=T)

    return data, S_true

def normalize_covariance(cov_matrix):
    variances = np.diagonal(cov_matrix)
    stddevs = np.sqrt(variances)
    diagonal_inv = np.diag(1 / stddevs)
    corr_matrix = diagonal_inv @ cov_matrix @ diagonal_inv
    return corr_matrix

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

def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 1e-2, log=True)
    beta = trial.suggest_float('beta', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.99)
    pinv_rcond = trial.suggest_float('pinv_rcond', 1e-4, 1e-2, log=True)
    eigval_threshold = trial.suggest_float('eigval_threshold', 1e-8, 1e-4, log=True)

    P = 1
    C = 1
    N = 3
    T = 2000

    data, S_true = generate_data(N, T, pinv_rcond)
    _, nses, _ = online_graph_learning(data, P, C, alpha, beta, gamma, pinv_rcond, eigval_threshold, S_true)
    
    return nses[-1]

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)
    
    best_params = study.best_params
    best_nse = study.best_value
    print(f"Best parameters: {best_params}")
    print(f"Best NSE: {best_nse}")

    alpha = best_params['alpha']
    beta = best_params['beta']
    gamma = best_params['gamma']
    pinv_rcond = best_params['pinv_rcond']
    eigval_threshold = best_params['eigval_threshold']
    
    P = 1
    C = 1
    N = 3
    T = 2000

    data, S_true = generate_data(N, T, pinv_rcond)
    S_hat, nses, f_list = online_graph_learning(data, P, C, alpha, beta, gamma, pinv_rcond, eigval_threshold, S_true)

    plt.figure(figsize=(8, 6))
    plt.semilogy(nses)
    plt.xlabel('Iteration')
    plt.ylabel('NSE (log scale)')
    plt.title('Online Graph Learning Convergence')
    plt.grid(True)
    plt.show()