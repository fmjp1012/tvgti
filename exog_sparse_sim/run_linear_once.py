import os
import shutil
import datetime

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/fmjp/Desktop/lab/simu/tvgti')

from exog_sparse_sim.data_gen import generate_linear_Y_with_exog
from exog_sparse_sim.models.pp_exog import PPExogenousSEM
from models.tvgti_pc.time_varing_sem import TimeVaryingSEM as PCSEM


def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family="serif")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 15

    N = 20
    T = 1000
    sparsity = 0.6
    max_weight = 0.5
    std_e = 0.05
    seed = 3
    np.random.seed(seed)

    S_series, B_true, U, Y = generate_linear_Y_with_exog(
        N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e,
        s_type="random", b_min=0.5, b_max=1.0, u_dist="uniform01"
    )

    S0 = np.zeros((N, N)); b0 = np.ones(N)
    r = 50; q = 5; rho = 1e-3; mu_lambda = 0.05
    pp = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
    S_hat_list, _ = pp.run(Y, U)

    # PC/CO/SGD baselines
    X = Y
    S0_pc = np.zeros((N, N))
    lambda_reg = 1e-3; alpha = 1e-2; beta = 1e-2; gamma = 0.9; P = 1; C = 1
    pc = PCSEM(N, S0_pc, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_baseline")
    estimates_pc, _ = pc.run(X)
    beta_co = 0.02
    co = PCSEM(N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C, show_progress=False, name="co_baseline")
    estimates_co, _ = co.run(X)
    beta_sgd = 0.0269
    sgd = PCSEM(N, S0_pc, lambda_reg, alpha, beta_sgd, 0.0, 0, C, show_progress=False, name="sgd_baseline")
    estimates_sgd, _ = sgd.run(X)

    err_pp = [
        (np.linalg.norm(S_hat_list[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
        for t in range(T)
    ]
    err_pc = [
        (np.linalg.norm(estimates_pc[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
        for t in range(T)
    ]
    err_co = [
        (np.linalg.norm(estimates_co[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
        for t in range(T)
    ]
    err_sgd = [
        (np.linalg.norm(estimates_sgd[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
        for t in range(T)
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(err_co, color='blue', label='Correction Only')
    plt.plot(err_pc, color='limegreen', label='Prediction Correction')
    plt.plot(err_sgd, color='cyan', label='SGD')
    plt.plot(err_pp, color='red', label='Proposed (PP)')
    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('t')
    plt.ylabel('NSE')
    plt.grid(True, which='both')
    plt.legend()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    notebook_filename = os.path.basename(__file__)
    filename = (f'timestamp{timestamp}_result_N{N}_notebook_filename{notebook_filename}_'
                f'T{T}_seed{seed}_r{r}_q{q}_rho{rho}_mulambda{mu_lambda}.png')
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


