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
# sgd手法用のモデルをインポート
from models.tvgti_pc_nonsparse import TimeVaryingSEM as TimeVaryingSEM_sgd_NONSPARSE

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
N: int = 30
T: int = 3000
sparsity: float = 0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True

# sgd手法固有の固定パラメータ
P: int = 0
C: int = 1
gamma: float = 0

seed: int = 3
np.random.seed(seed)

# TV-SEMシミュレーション
S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)

snr_before: float = calc_snr(S_series[0])
print("Before scaling, SNR =", snr_before)

# 初期値の設定
S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
S_0 = S_0 / norm(S_0)

#----------------------------------------------------
# sgd手法実行関数定義
def run_tv_sem_sgd(beta_sgd: float) -> List[np.ndarray]:
    """Run the sgd method with specified hyperparams."""
    tv_sem_sgd = TimeVaryingSEM_sgd_NONSPARSE(
        N, S_0, 0, beta_sgd, gamma, P, C, name="sgd"
    )
    # モデル実行：cost_valuesは今回は利用しないので無視
    estimates_sgd, _ = tv_sem_sgd.run(X)
    return estimates_sgd

#-----------------------------------------------------------
# Optuna で sgd 手法のハイパーパラメータをチューニング
#-----------------------------------------------------------

def objective(trial: optuna.trial.Trial) -> float:
    """
    目的関数：最終時刻でのNSEを評価指標として最小化する
    NSE_t = ||S_hat(t) - S_series(t)||^2 / ||S_0 - S_series(t)||^2
    """
    # 1) ハイパーパラメータのサンプリング
    #    αとβₚ𝚌を対数スケールでサンプリング
    beta_sgd_suggested = trial.suggest_float("beta_sgd", 1e-6, 3, log=True)

    # 2) モデルを作成して実行
    estimates_sgd = run_tv_sem_sgd(beta_sgd=beta_sgd_suggested)

    # 3) 評価指標の計算（最終時刻のNSE）
    final_estimate = estimates_sgd[-1]
    final_true = S_series[-1]
    final_nse = (norm(final_estimate - final_true) ** 2) / (norm(S_0 - final_true) ** 2)
    
    return final_nse

# Optuna で探索
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)  # お好みでトライアル数を設定

print("Study best trial:")
best_trial = study.best_trial
print("  Params:", best_trial.params)
print("  Value (final NSE):", best_trial.value)

# ここで得られたベストパラメータを使って、再度モデルを走らせる
best_beta_sgd = best_trial.params["beta_sgd"]

print(f"Best Hyperparams => beta_sgd={best_beta_sgd}, gamma={gamma}, P={P}, C={C}")

#----------------------------------------------------
# チューニング済みパラメータで最終実行（sgd手法）
#----------------------------------------------------

estimates_sgd_tuned = run_tv_sem_sgd(beta_sgd=best_beta_sgd)

# チューニング後のNSEを計算
error_sgd_tuned = []
for i, estimate in enumerate(estimates_sgd_tuned):
    val = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
    error_sgd_tuned.append(val)

# プロット
plt.figure(figsize=(10, 6))
plt.plot(error_sgd_tuned, color='limegreen', label='SGD (Tuned sgd)')
plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('NSE')
plt.grid(True, which="both")
plt.legend()

timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
notebook_filename: str = os.path.basename(__file__)

filename: str = (
    f'timestamp{timestamp}_'
    f'result_N{N}_'
    f'notebook_filename{notebook_filename}_'
    f'seed{seed}_'
    f'beta_sgd{best_beta_sgd}_'
    f'gamma{gamma}_'
    f'P{P}_'
    f'C{C}.png'
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
