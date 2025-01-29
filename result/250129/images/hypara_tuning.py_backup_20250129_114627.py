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
from models.tvgti_pc_nonsparse import TimeVaryingSEM as TimeVaryingSEM_PC_NONSPARSE
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED

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

#----------------------------------------------------
# メソッドごとの実行スイッチ
run_pc_flag: bool = False     # Prediction Correction
run_co_flag: bool = False     # Correction Only
run_sgd_flag: bool = False    # SGD
run_pp_flag: bool = True      # Proposed (we will tune this)
#----------------------------------------------------

# パラメータの設定
N: int = 30
T: int = 10000
sparsity: float = 0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 4
S_is_symmetric: bool = True

seed: int = 3
np.random.seed(seed)

# TV-SEMシミュレーション
S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)

snr_before: float = calc_snr(S_series[0])
print("Before scaling, SNR =", snr_before)

# オンラインTV-SEMパラメータ（pc, co, sgd は固定とします）
P: int = 1
C: int = 1
gamma: float = 0.999
alpha: float = 0.015
beta_pc: float = 0.015
beta_co: float = 0.02
beta_sgd: float = 0.02

# 初期値の設定
S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
S_0 = S_0 / norm(S_0)

# pp 手法のデフォルトパラメータ（後でOptunaで上書きされる）
r_def: int = 30  # window size
q_def: int = 10  # number of processors
rho_def: float = 2.4  # 試行回数の設定
mu_lambda_def: float = 1.0

# 実行関数定義
def run_tv_sem_pp(r: int, q: int, rho: float, mu_lambda: float) -> List[np.ndarray]:
    """Run the pp method with specified hyperparams."""
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(
        N, S_0, r, q, rho, mu_lambda, name="pp"
    )
    estimates_pp = tv_sem_pp.run(X)
    return estimates_pp

#-----------------------------------------------------------
# Optuna で pp 手法のハイパーパラメータをチューニング
#-----------------------------------------------------------

def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function to minimize. We'll compute the average NSE across all time steps t.
    """
    # 1) ハイパーパラメータのサンプリング
    #    適当に検索範囲を設定していますので、必要に応じて変更してください
    r_suggested = trial.suggest_int("r", 5, 50, step=5)     # 5,10,15,20, ..., 50
    q_suggested = trial.suggest_int("q", 5, 50, step=5)     # 5,10,15,20, ..., 50
    rho_suggested = trial.suggest_float("rho", 0.1, 5.0, log=True)
    mu_lambda_suggested = trial.suggest_float("mu_lambda", 0.001, 1.999, log=False)

    # 2) モデルを作成して実行
    estimates_pp = run_tv_sem_pp(
        r=r_suggested,
        q=q_suggested,
        rho=rho_suggested,
        mu_lambda=mu_lambda_suggested
    )

    # 3) 評価指標を計算 (ここでは平均 NSE)
    #    NSE_t = ||S_hat(t) - S_series(t)||^2 / ||S_0 - S_series(t)||^2
    #    → 全時刻 t の NSE を平均化
    #    必要に応じて別の指標に変更可
    nse_list = []
    for i, estimate in enumerate(estimates_pp):
        numerator = norm(estimate - S_series[i])**2
        denominator = norm(S_0 - S_series[i])**2
        nse_list.append(numerator / denominator)
    avg_nse = np.mean(nse_list)

    return avg_nse

# Optuna で探索
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # お好みでトライアル数を設定

print("Study best trial:")
best_trial = study.best_trial
print("  Params:", best_trial.params)
print("  Value (avg NSE):", best_trial.value)

# ここで得られたベストパラメータを使って、再度モデルを走らせる
best_r = best_trial.params["r"]
best_q = best_trial.params["q"]
best_rho = best_trial.params["rho"]
best_mu_lambda = best_trial.params["mu_lambda"]

print(f"Best Hyperparams => r={best_r}, q={best_q}, rho={best_rho}, mu_lambda={best_mu_lambda}")

#----------------------------------------------------
# チューニング済みパラメータで最終実行
#----------------------------------------------------

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
    f'result_N{N}_'
    f'notebook_filename{notebook_filename}_'
    f'T{T}_'
    f'maxweight{max_weight}_'
    f'variancee{variance_e}_'
    f'K{K}_'
    f'Sissymmetric{S_is_symmetric}_'
    f'seed{seed}_'
    f'alpha{alpha}_'
    f'r{best_r}_'
    f'q{best_q}_'
    f'rho{best_rho}_'
    f'mu_lambda{best_mu_lambda}_'
    f'timestamp{timestamp}.png'
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
