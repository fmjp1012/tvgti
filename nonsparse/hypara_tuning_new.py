import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import shutil
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
N: int = 5
T: int = 100
sparsity: float = 0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True

r_fixed = 1
q_fixed = 1
mu_lambda_fixed = 1

seed: int = 3
np.random.seed(seed)

# TV-SEMシミュレーション
S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)

snr_before: float = calc_snr(S_series[0])
print("Before scaling, SNR =", snr_before)

# 初期値の設定
S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
S_0 = S_0 / norm(S_0)

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
    目的関数を、最終時刻1点のNSEではなく、
    シミュレーション後半のウィンドウにおける平均NSEとそのばらつきを組み合わせた指標に変更する例。
    """
    # 1) ハイパーパラメータのサンプリング
    r_suggested = r_fixed
    q_suggested = q_fixed
    mu_lambda_suggested = trial.suggest_float("mu_lambda", 1e-6, 2 - 1e-6, log=False)
    rho_suggested = trial.suggest_float("rho", 1e-6, 3, log=False)
    
    # 2) モデルの実行
    estimates_pp = run_tv_sem_pp(
        r=r_suggested,
        q=q_suggested,
        rho=rho_suggested,
        mu_lambda=mu_lambda_suggested
    )
    
    # 3) 各時刻でのNSEを計算
    nse_list = []
    for i, estimate in enumerate(estimates_pp):
        nse_val = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        nse_list.append(nse_val)
    
    # 4) 評価ウィンドウの設定 (例: 最後20%の時刻)
    window_size = int(0.2 * T)  # Tは総時刻数
    window_nse = nse_list[-window_size:]
    
    # 5) ウィンドウ内の平均NSEと標準偏差を計算
    avg_nse = np.mean(window_nse)
    std_nse = np.std(window_nse)
    
    # 6) 目的関数として、平均NSEにばらつきのペナルティを加える（lambda_penaltyで重み付け）
    lambda_penalty = 1.0  # この値は評価の重み付けとして調整可能
    objective_value = avg_nse + lambda_penalty * std_nse
    
    return objective_value

# Optuna で探索
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # 100 -> 10に変更

print("Study best trial:")
best_trial = study.best_trial
print("  Params:", best_trial.params)
print("  Value (avg NSE):", best_trial.value)

# ここで得られたベストパラメータを使って、再度モデルを走らせる
best_r = r_fixed
best_q = q_fixed

# best_r = best_trial.params["r"]
# best_q = best_trial.params["q"]
best_mu_lambda = best_trial.params["mu_lambda"]
best_rho = best_trial.params["rho"]
# best_mu_lambda = mu_lambda_fixed

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
    f'timestamp{timestamp}_'
    f'result_N{N}_'
    f'notebook_filename{notebook_filename}_'
    # f'T{T}_'
    # f'maxweight{max_weight}_'
    # f'variancee{variance_e}_'
    # f'K{K}_'
    # f'Sissymmetric{S_is_symmetric}_'
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
shutil.copy(__file__, copy_ipynb_path)  # notebook_filename -> __file__に変更
print(f"Notebook file copied to: {copy_ipynb_path}")
