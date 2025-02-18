import shutil
import sys
import os
import datetime
from typing import List, Tuple

import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from utils import *
from models.tvgti_pc_nonsparse import TimeVaryingSEM as TimeVaryingSEM_PC_NONSPARSE
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED

# --- プロットの設定 ---
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

# --- 手法ごとの実行スイッチ ---
run_pc_flag: bool = False     # Prediction Correction
run_co_flag: bool = False     # Correction Only
run_sgd_flag: bool = False    # SGD
run_pp_flag: bool = True      # Proposed

# --- シミュレーション全体のパラメータ ---
N: int = 10             # ノード数
T: int = 10000          # 時系列長
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True
seed: int = 3
np.random.seed(seed)

# --- オンラインTV-SEMのパラメータ ---
P: int = 1
C: int = 1
gamma: float = 0.999
alpha: float = 0.015
beta_pc: float = 0.015
beta_co: float = 0.02
beta_sgd: float = 0.02

# --- Proposed手法のパラメータ ---
r: int = 4   # window size
q: int = 10  # number of processors
rho: float = 0.15
mu_lambda: float = 1

# --- 接続率（リンク数の割合）を変化させる ---
# ここでは10%～90%の9段階でシミュレーション
link_ratios = np.linspace(0.1, 0.9, 9)  # 例：0.1なら全体の10%のリンク
link_percentage_values = link_ratios * 100  # プロット用に%

# --- 各手法の最終NSEの平均値を格納するリスト ---
final_NSE_pc = []
final_NSE_co = []
final_NSE_sgd = []
final_NSE_pp = []

# --- 結果画像の保存先 ---
today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_dir: str = os.path.join('.', 'result', today_str, 'images')
os.makedirs(save_dir, exist_ok=True)

# --- 試行回数 ---
num_trials = 100

# --- 1回の試行を実行する関数 ---
def run_simulation_trial(sparsity: float) -> Tuple[float, float, float, float]:
    """
    1回の試行を実行し、各手法の最終NSEを返す。
    該当手法が無効の場合はnp.nanを返す。
    """
    # 真の構造行列系列 S_series と観測データ X の生成
    S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)
    
    # 初期値 S_0 の生成
    S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_0 = S_0 / norm(S_0)
    
    # --- 各手法のモデルのインスタンス化 ---
    tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, name="pc")
    tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, name="co")
    tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, name="sgd")
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, name="pp")
    
    # --- 各手法の実行関数をリスト化 ---
    job_list = []
    if run_pc_flag:
        job_list.append(delayed(tv_sem_pc.run)(X))
    if run_co_flag:
        job_list.append(delayed(tv_sem_co.run)(X))
    if run_sgd_flag:
        job_list.append(delayed(tv_sem_sgd.run)(X))
    if run_pp_flag:
        job_list.append(delayed(tv_sem_pp.run)(X))
    
    # 並列実行（内部の並列処理との二重並列に注意）
    results = Parallel(n_jobs=4)(job_list)
    
    # --- 各手法の最終NSEを計算 ---
    idx_result: int = 0
    final_pc = np.nan
    final_co = np.nan
    final_sgd = np.nan
    final_pp = np.nan

    if run_pc_flag:
        estimates_pc, _ = results[idx_result]
        num_iter = min(len(estimates_pc), len(S_series))
        error_curve_pc = [(norm(estimates_pc[i] - S_series[i])**2) / (norm(S_0 - S_series[i])**2) 
                           for i in range(num_iter)]
        final_pc = error_curve_pc[-1]
        idx_result += 1

    if run_co_flag:
        estimates_co, _ = results[idx_result]
        num_iter = min(len(estimates_co), len(S_series))
        error_curve_co = [(norm(estimates_co[i] - S_series[i])**2) / (norm(S_0 - S_series[i])**2) 
                           for i in range(num_iter)]
        final_co = error_curve_co[-1]
        idx_result += 1

    if run_sgd_flag:
        estimates_sgd, _ = results[idx_result]
        num_iter = min(len(estimates_sgd), len(S_series))
        error_curve_sgd = [(norm(estimates_sgd[i] - S_series[i])**2) / (norm(S_0 - S_series[i])**2) 
                           for i in range(num_iter)]
        final_sgd = error_curve_sgd[-1]
        idx_result += 1

    if run_pp_flag:
        estimates_pp = results[idx_result]
        num_iter = min(len(estimates_pp), len(S_series))
        error_curve_pp = [(norm(estimates_pp[i] - S_series[i])**2) / (norm(S_0 - S_series[i])**2) 
                          for i in range(num_iter)]
        final_pp = error_curve_pp[-1]
        idx_result += 1

    return final_pc, final_co, final_sgd, final_pp

# --- 各接続率に対するシミュレーション ---
for sparsity in link_ratios:
    print(f"Link ratio {sparsity*100:.1f}%: Running {num_trials} trials...")
    # 各試行の最終NSEを格納するリスト
    trial_NSE_pc = []
    trial_NSE_co = []
    trial_NSE_sgd = []
    trial_NSE_pp = []
    
    for trial in range(num_trials):
        final_pc, final_co, final_sgd, final_pp = run_simulation_trial(sparsity)
        if run_pc_flag:
            trial_NSE_pc.append(final_pc)
        if run_co_flag:
            trial_NSE_co.append(final_co)
        if run_sgd_flag:
            trial_NSE_sgd.append(final_sgd)
        if run_pp_flag:
            trial_NSE_pp.append(final_pp)
    
    # 平均値を計算（各手法ごと）
    avg_NSE_pc = np.nanmean(trial_NSE_pc) if run_pc_flag else np.nan
    avg_NSE_co = np.nanmean(trial_NSE_co) if run_co_flag else np.nan
    avg_NSE_sgd = np.nanmean(trial_NSE_sgd) if run_sgd_flag else np.nan
    avg_NSE_pp = np.nanmean(trial_NSE_pp) if run_pp_flag else np.nan

    final_NSE_pc.append(avg_NSE_pc)
    final_NSE_co.append(avg_NSE_co)
    final_NSE_sgd.append(avg_NSE_sgd)
    final_NSE_pp.append(avg_NSE_pp)

# --- 結果のプロット（最終時刻の平均NSE vs リンク割合） ---
plt.figure(figsize=(10, 6))
if run_pc_flag:
    plt.plot(link_percentage_values, final_NSE_pc, marker='o', color='limegreen', label='Prediction Correction')
if run_co_flag:
    plt.plot(link_percentage_values, final_NSE_co, marker='s', color='blue', label='Correction Only')
if run_sgd_flag:
    plt.plot(link_percentage_values, final_NSE_sgd, marker='^', color='cyan', label='SGD')
if run_pp_flag:
    plt.plot(link_percentage_values, final_NSE_pp, marker='d', color='red', label='Proposed')

plt.yscale('log')
plt.xlabel('Link Percentage (%)')
plt.ylabel('Average Final NSE')
plt.grid(True, which='both')
plt.legend()

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
notebook_filename: str = os.path.basename(__file__) if '__file__' in globals() else 'simulation.py'
filename_final: str = (
    f'result_N{N}_'
    f'{notebook_filename}_'
    f'T{T}_'
    f'maxweight{max_weight}_'
    f'variancee{variance_e}_'
    f'K{K}_'
    f'Sissymmetric{S_is_symmetric}_'
    f'seed{seed}_'
    f'P{P}_'
    f'C{C}_'
    f'gammma{gamma}_'
    f'alpha{alpha}_'
    f'betapc{beta_pc}_'
    f'betaco{beta_co}_'
    f'betasgd{beta_sgd}_'
    f'r{r}_'
    f'q{q}_'
    f'rho{rho}_'
    f'mu_lambda{mu_lambda}_'
    f'timestamp{timestamp}.png'
)

plt.savefig(os.path.join(save_dir, filename_final))
plt.show()

# --- コードファイルのバックアップ ---
copy_script_path: str = os.path.join(save_dir, f"{notebook_filename}_backup_{timestamp}.py")
try:
    shutil.copy(notebook_filename, copy_script_path)
    print(f"Script file copied to: {copy_script_path}")
except Exception as e:
    print("Backup failed:", e)
