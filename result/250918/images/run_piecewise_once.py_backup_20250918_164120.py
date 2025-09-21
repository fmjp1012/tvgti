import os
import shutil
import datetime

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import cvxpy as cp

import sys
sys.path.append('/Users/fmjp/Desktop/lab/simu/tvgti')

from exog_sparse_sim.data_gen import generate_piecewise_Y_with_exog
from exog_sparse_sim.models.pp_exog import PPExogenousSEM
from models.tvgti_pc.time_varing_sem import TimeVaryingSEM as PCSEM
from models.tvgti_pc.time_varing_sem import TimeVaryingSEMWithL1Correction as PCSEM_L1C
from utils import elimination_matrix_hh, duplication_matrix_hh


def main():
    # プロット設定
    plt.rc('text', usetex=True)
    plt.rc('font', family="serif")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 15

    # パラメータ
    N = 20
    T = 1000
    sparsity = 0.7
    max_weight = 0.5
    std_e = 0.05
    K = 4
    seed = 3
    np.random.seed(seed)

    # 外部設定の読み込み（JSON）
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="ハイパラ設定JSONのパス")
    args, _ = parser.parse_known_args()

    # 生成
    S_series, B_true, U, Y = generate_piecewise_Y_with_exog(
        N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
        s_type="random", b_min=0.5, b_max=1.0, u_dist="uniform01"
    )

    # 既定ハイパラ（fallback）
    r = 50; q = 5; rho = 1e-3; mu_lambda = 0.05
    lambda_reg = 1e-3; alpha = 1e-2; beta = 1e-2; gamma = 0.9; P = 1; C = 1
    beta_co = 0.02
    beta_sgd = 0.0269

    # JSON 設定で上書き
    if args.config is not None and os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        pp_cfg = cfg.get('pp', {})
        r = pp_cfg.get('r', r)
        q = pp_cfg.get('q', q)
        rho = pp_cfg.get('rho', rho)
        mu_lambda = pp_cfg.get('mu_lambda', mu_lambda)

        pc_cfg = cfg.get('pc', {})
        lambda_reg = pc_cfg.get('lambda_reg', lambda_reg)
        alpha = pc_cfg.get('alpha', alpha)
        beta = pc_cfg.get('beta', beta)
        gamma = pc_cfg.get('gamma', gamma)
        P = pc_cfg.get('P', P)
        C = pc_cfg.get('C', C)

        co_cfg = cfg.get('co', {})
        beta_co = co_cfg.get('beta_co', beta_co)

        sgd_cfg = cfg.get('sgd', {})
        beta_sgd = sgd_cfg.get('beta_sgd', beta_sgd)

    # PP（本番実行）
    S0 = np.zeros((N, N))
    b0 = np.ones(N)
    pp = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
    S_hat_list, _ = pp.run(Y, U)

    # PC/CO/SGD baseline
    X = Y
    S0_pc = np.zeros((N, N))
    pc = PCSEM(N, S0_pc, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_baseline")
    estimates_pc, _ = pc.run(X)
    co = PCSEM(N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C, show_progress=False, name="co_baseline")
    estimates_co, _ = co.run(X)
    sgd = PCSEM(N, S0_pc, lambda_reg, alpha, beta_sgd, 0.0, 0, C, show_progress=False, name="sgd_baseline")
    estimates_sgd, _ = sgd.run(X)

    # 新手法: PC + L1 correction
    pc_l1c = PCSEM_L1C(N, S0_pc, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_l1corr")
    estimates_pc_l1c, _ = pc_l1c.run(X)

    # 誤差プロット
    err_pp = [np.linalg.norm(S_hat_list[t] - S_series[t], ord='fro') for t in range(T)]
    err_pc = [np.linalg.norm(estimates_pc[t] - S_series[t], ord='fro') for t in range(T)]
    err_co = [np.linalg.norm(estimates_co[t] - S_series[t], ord='fro') for t in range(T)]
    err_sgd = [np.linalg.norm(estimates_sgd[t] - S_series[t], ord='fro') for t in range(T)]
    err_pc_l1c = [np.linalg.norm(estimates_pc_l1c[t] - S_series[t], ord='fro') for t in range(T)]

    # オフライン最適（全データでのコスト関数をcvxpyで最小化）
    def compute_offline_mean_error_cvx(X_mat: np.ndarray, S_series_true, gamma_val: float) -> float:
        N_loc = X_mat.shape[0]
        T_loc = X_mat.shape[1]
        D_h = duplication_matrix_hh(N_loc).tocsc()
        E_h = elimination_matrix_hh(N_loc).tocsc()
        D_h_T = D_h.T
        l_loc = N_loc * (N_loc - 1) // 2

        # 累積Q, 累積sigma
        Q_sum = np.zeros((l_loc, l_loc))
        sigma_sum = np.zeros(l_loc)
        Sigma_t = np.zeros((N_loc, N_loc))
        for t in range(T_loc):
            x = X_mat[:, t].reshape(-1, 1)
            Sigma_t = gamma_val * Sigma_t + (1 - gamma_val) * (x @ x.T)
            Sigma_kron_I = np.kron(Sigma_t, np.eye(N_loc))
            Q_t = (D_h_T @ Sigma_kron_I @ D_h)
            sigma_t = E_h @ Sigma_t.flatten()
            Q_sum += Q_t
            sigma_sum += sigma_t

        # cvxpyで min 0.5*s^T Q_sum s - 2*sigma_sum^T s
        s = cp.Variable(l_loc)
        # 数値安定化のため微小リッジを追加
        ridge = 1e-8 * np.eye(l_loc)
        objective = 0.5 * cp.quad_form(s, Q_sum + ridge) - 2.0 * sigma_sum @ s
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
        if s.value is None:
            prob.solve(solver=cp.SCS, eps=1e-5, verbose=False)
        s_star = s.value
        # 復元
        S_flat = (D_h @ s_star).reshape(-1)
        S_mat = np.zeros((N_loc, N_loc))
        idx = 0
        for i in range(N_loc):
            for j in range(i + 1, N_loc):
                S_mat[i, j] = S_flat[idx]
                S_mat[j, i] = S_flat[idx]
                idx += 1
        np.fill_diagonal(S_mat, 0.0)
        # 平均フロベニウス誤差
        errors = [np.linalg.norm(S_mat - S_series_true[t], ord='fro') for t in range(T_loc)]
        return float(np.mean(errors))

    plt.figure(figsize=(10, 6))
    plt.plot(err_co, color='blue', label='Correction Only')
    plt.plot(err_pc, color='limegreen', label='Prediction Correction')
    plt.plot(err_sgd, color='cyan', label='SGD')
    plt.plot(err_pp, color='red', label='Proposed (PP)')
    plt.plot(err_pc_l1c, color='magenta', label='PC + L1 correction')
    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('t')
    plt.ylabel('Frobenius error')
    plt.grid(True, which='both')
    # オフライン平均誤差（全データcvxpy最適化）を横線（点線）で追加
    pc_offline_err = compute_offline_mean_error_cvx(X, S_series, gamma)
    co_offline_err = compute_offline_mean_error_cvx(X, S_series, gamma)
    sgd_offline_err = compute_offline_mean_error_cvx(X, S_series, 0.0)
    plt.axhline(y=pc_offline_err, color='limegreen', linestyle='--', alpha=0.7, label='PC offline (mean)')
    plt.axhline(y=co_offline_err, color='blue', linestyle='--', alpha=0.7, label='CO offline (mean)')
    plt.axhline(y=sgd_offline_err, color='cyan', linestyle='--', alpha=0.7, label='SGD offline (mean)')
    plt.legend()

    # 保存
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    notebook_filename = os.path.basename(__file__)
    filename = (f'timestamp{timestamp}_result_N{N}_notebook_filename{notebook_filename}_'
                f'T{T}_K{K}_seed{seed}_r{r}_q{q}_rho{rho}_mulambda{mu_lambda}.png')
    print(filename)
    today_str = datetime.datetime.now().strftime('%y%m%d')
    save_path = os.path.join('./result', today_str, 'images')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename))
    plt.show()

    copy_py_path = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
    shutil.copy(__file__, copy_py_path)
    print(f"Script file copied to: {copy_py_path}")


if __name__ == "__main__":
    main()


