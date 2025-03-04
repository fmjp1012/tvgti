import shutil
import sys
import os
import datetime
from typing import List, Tuple, Dict

import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager

import optuna  # <-- Import Optuna

from utils import *
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED

#----------------------------------------------------
# プロット設定
plt.rc('text', usetex=True)
plt.rc('font', family="serif")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.minor.width"] = 1.0
plt.rcParams["ytick.minor.width"] = 1.0
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["xtick.minor.size"] = 5
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["font.size"] = 15

# パラメータの設定
N: int = 10
T: int = 10000
sparsity: float = 0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True

r_fixed = 2
q_fixed = 1

seed: int = 3
np.random.seed(seed)

#----------------------------------------------------
# 初回シミュレーション（初期確認用）
S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)
snr_before: float = calc_snr(S_series[0])
print("Before scaling, SNR =", snr_before)

# 初期値の設定
S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
S_0 = S_0 / norm(S_0)

#----------------------------------------------------
# 実行関数定義（グローバル変数 S_series, X, S_0 に依存）
def run_tv_sem_pp(r: int, q: int, rho: float, mu_lambda: float) -> List[np.ndarray]:
    """指定ハイパーパラメータで pp 手法を実行する関数"""
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(
        N, S_0, r, q, rho, mu_lambda, name="pp"
    )
    estimates_pp = tv_sem_pp.run(X)
    return estimates_pp

#----------------------------------------------------
# 1回分のシミュレーション試行（新たなシミュレーションデータを生成）
def run_simulation_trial(r: int, q: int, rho: float, mu_lambda: float) -> float:
    # 毎回新しいシミュレーションデータを生成
    S_series_new, X_new = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)
    S0_new = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S0_new = S0_new / norm(S0_new)
    
    # run_tv_sem_pp がグローバル変数を利用しているため更新
    global S_series, X, S_0
    S_series = S_series_new
    X = X_new
    S_0 = S0_new
    
    # モデルの実行
    estimates_pp = run_tv_sem_pp(r, q, rho, mu_lambda)
    final_estimate = estimates_pp[-1]
    final_true = S_series[-1]
    final_nse = (norm(final_estimate - final_true) ** 2) / (norm(S_0 - final_true) ** 2)
    return final_nse

#----------------------------------------------------
# Optuna の目的関数定義（5回分の試行を並列実行して平均NSEを計算）
def objective(trial: optuna.trial.Trial) -> float:
    """
    目的関数：各トライアルで5回新たなシミュレーションデータを生成し、
    各試行の最終時刻のNSEの平均値を返す。
    """
    # ハイパーパラメータのサンプリング（r, q は固定）
    r_suggested = r_fixed
    q_suggested = q_fixed
    rho_suggested = trial.suggest_float("rho", 1e-6, 0.1, log=False)
    mu_lambda_suggested = trial.suggest_float("mu_lambda", 1e-6, 1, log=False)

    n_runs = 5
    # joblib で5回の試行を並列実行
    final_nse_list = Parallel(n_jobs=n_runs)(
        delayed(run_simulation_trial)(r_suggested, q_suggested, rho_suggested, mu_lambda_suggested)
        for _ in range(n_runs)
    )
    avg_final_nse = np.mean(final_nse_list)
    return avg_final_nse

#----------------------------------------------------
# Optuna でハイパーパラメータ探索
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000)  # トライアル数は必要に応じて変更

print("Study best trial:")
best_trial = study.best_trial
print("  Params:", best_trial.params)
print("  Value (avg NSE):", best_trial.value)

best_r = r_fixed
best_q = q_fixed
best_mu_lambda = best_trial.params["mu_lambda"]
best_rho = best_trial.params["rho"]

print(f"Best Hyperparams => r={best_r}, q={best_q}, rho={best_rho}, mu_lambda={best_mu_lambda}")

#----------------------------------------------------
# チューニング済みパラメータで最終実行
estimates_pp_tuned = run_tv_sem_pp(
    r=best_r,
    q=best_q,
    rho=best_rho,
    mu_lambda=best_mu_lambda
)

# チューニング後のNSEを計算
error_pp_tuned = []
for i, estimate in enumerate(estimates_pp_tuned):
    val = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
    error_pp_tuned.append(val)

#----------------------------------------------------
# プロット
plt.figure(figsize=(10, 6))
plt.plot(error_pp_tuned, color='red', label='Proposed (Tuned pp)')
plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('NSE')
plt.grid(True, "both")
plt.legend()

timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
notebook_filename: str = os.path.basename(__file__)

filename: str = (
    f'timestamp{timestamp}_'
    f'result_N{N}_'
    f'notebook_filename{notebook_filename}_'
    f'seed{seed}_'
    f'r{best_r}_'
    f'q{best_q}_'
    f'rho{best_rho}_'
    f'mu_lambda{best_mu_lambda}.png'
)
print(filename)
today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
