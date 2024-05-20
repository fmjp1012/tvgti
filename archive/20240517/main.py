import numpy as np
import matplotlib.pyplot as plt
from utils import *
from cvx.sdp import *

pred_grad_f_list = []
pred_grad_ts_f_list = []
cor_grad_f_list = []
s_hat_list = []
theta_hat_list = []
f_list = []

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
    eigvals = np.maximum(eigvals, 0)
    S = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return mat_to_vech(S)

def normalize_covariance(cov_matrix):
    variances = np.diagonal(cov_matrix)
    stddevs = np.sqrt(variances)
    diagonal_inv = np.diag(1 / stddevs)
    corr_matrix = diagonal_inv @ cov_matrix @ diagonal_inv
    return corr_matrix

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


def online_graph_learning(data, P, C, alpha, beta, gamma, S_true):
    N = data.shape[1]
    s_hat = mat_to_vech(S_true)
    Theta_hat = np.linalg.inv(S_true)
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
        f = -np.log(np.linalg.det(S_hat) + 1e-6) + np.trace(S_hat @ Theta_hat)
        f_list.append(f)

    return vech_to_mat(s_hat), nses, f_list

def generate_data(N, T, seed=24):
    np.random.seed(seed)
    X = np.random.rand(N, N)
    Theta_true = normalize_covariance(X @ X.T)

    S_true = np.linalg.inv(Theta_true)
    data = np.random.multivariate_normal(mean=np.zeros(N), cov=Theta_true, size=T)

    return data, S_true

if __name__ == "__main__":

    # pred_grad_f_list = []
    # pred_grad_ts_f_list = []
    # cor_grad_f_list = []
    # s_hat_list = []
    # theta_hat_list = []

    N = 10
    T = 20000

    P = 1
    C = 1
    # alpha = 0.0008809864430585985
    # beta = 0.0007182262456343882
    # gamma = 0.5167807826874304
    # pinv_rcond = 0.0000022185520940586874
    alpha = 0.001
    beta = 0.001
    gamma = 0.99

    data, S_true = generate_data(N, T, seed=1000)

    S_hat, nses, f_list = online_graph_learning(data, P, C, alpha, beta, gamma, S_true)

    plt.figure(figsize=(8, 6))
    plt.semilogy(nses)
    plt.xlabel('Iteration')
    plt.ylabel('NSE (log scale)')
    plt.title('Online Graph Learning Convergence')
    plt.grid(True)
    plt.show()
data = {
    'pred_grad_f_list': pred_grad_f_list,
    'pred_grad_ts_f_list': pred_grad_ts_f_list,
    'cor_grad_f_list': cor_grad_f_list,
    's_hat_list': s_hat_list,
    'theta_hat_list': theta_hat_list
}

# サブプロットを作成
fig, axs = plt.subplots(5, 1, figsize=(10, 15))

# 各リストに対してプロットを作成
for i, (title, values) in enumerate(data.items()):
    axs[i].plot(values)
    axs[i].set_title(title)

# レイアウトを調整して表示
plt.tight_layout()
plt.show()
