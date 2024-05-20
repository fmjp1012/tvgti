import numpy as np
import matplotlib.pyplot as plt
import optuna


def predict(s_hat, Theta_hat, Theta_hat_prev, alpha, h, D):
    s = s_hat.copy()
    S_inv = np.linalg.inv(vech_to_mat(s_hat))
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
    eigvals = np.maximum(eigvals, 0.01)
    S = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return mat_to_vech(S)

def online_graph_learning(data, P, C, alpha, beta, gamma, S_true):
    N = data.shape[1]
    Theta_hat = np.cov(data[:100, :], rowvar=False)
    s_hat = mat_to_vech(np.linalg.inv(Theta_hat))
    Theta_hat_prev = Theta_hat
    nses = []
    D = duplication_matrix(N)

    for t in range(N, data.shape[0]):
        for _ in range(P):
            s_hat = predict(s_hat, Theta_hat, Theta_hat_prev, alpha, 1, D)

        x_t = data[t, :]
        Theta_hat_prev = Theta_hat
        Theta_hat = gamma * Theta_hat_prev + (1 - gamma) * np.outer(x_t, x_t)

        for _ in range(C):
            s_hat = correct(s_hat, Theta_hat, beta, D)

        S_hat = vech_to_mat(s_hat)
        nse = np.linalg.norm(S_hat - S_true, ord='fro') ** 2 / np.linalg.norm(S_true, ord='fro') ** 2
        nses.append(nse)
        f = -np.log(np.linalg.det(S_hat)) + np.trace(S_hat @ Theta_hat)

    return vech_to_mat(s_hat), nses

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

def generate_data(N, T, seed=24):
    np.random.seed(seed)
    X = np.random.rand(N, N)
    Theta_true = np.cov(X)

    eigvals, eigvecs = np.linalg.eigh(Theta_true)
    Theta_true = eigvecs @ np.diag([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) @ eigvecs.T

    S_true = np.linalg.inv(Theta_true)
    data = np.random.multivariate_normal(mean=np.zeros(N), cov=Theta_true, size=T)

    return data, S_true

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

    P = 1
    C = 1
    N = 10
    T = 2000

    data, S_true = generate_data(N, T)
    _, nses = online_graph_learning(data, P, C, alpha, beta, gamma, S_true)
    
    return nses[-1]

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    best_params = study.best_params
    best_nse = study.best_value
    print(f"Best parameters: {best_params}")
    print(f"Best NSE: {best_nse}")

    alpha = best_params['alpha']
    beta = best_params['beta']
    gamma = best_params['gamma']
    
    P = 1
    C = 1
    N = 10
    T = 2000

    data, S_true = generate_data(N, T)
    S_hat, nses = online_graph_learning(data, P, C, alpha, beta, gamma, S_true)

    plt.figure(figsize=(8, 6))
    plt.semilogy(nses)
    plt.xlabel('Iteration')
    plt.ylabel('NSE (log scale)')
    plt.title('Online Graph Learning Convergence')
    plt.grid(True)
    plt.show()