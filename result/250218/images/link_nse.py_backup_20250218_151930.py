import shutil
import sys
import os
import datetime
from typing import List, Tuple

import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from multiprocessing import Manager

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
run_pp_flag: bool = True     # Proposed

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

# オンラインTV-SEMのパラメータ
P: int = 1
C: int = 1
gamma: float = 0.999
alpha: float = 0.015
beta_pc: float = 0.015
beta_co: float = 0.02
beta_sgd: float = 0.02

# Proposed手法のパラメータ
r: int = 4   # window size
q: int = 10  # number of processors
rho: float = 0.15
mu_lambda: float = 1

# --- 接続率（リンク数の割合）を変化させる ---
# ここでは10%～100%の10段階でシミュレーション
link_ratios = np.linspace(0.1, 0.9, 9)  # 例：0.1なら全体の10%のリンク
link_percentage_values = link_ratios * 100  # プロット用に%

# 各手法の最終NSEを格納するリスト
final_NSE_pc = []
final_NSE_co = []
final_NSE_sgd = []
final_NSE_pp = []

# --- 各接続率に対するシミュレーション ---
for sparsity in link_ratios:
    # ※ここでの sparsity は「リンクが存在する割合」として扱う

    # 真の構造行列系列 S_series と観測データ X の生成
    S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)
    # （必要に応じて SNR の確認も可能）
    # print("SNR =", calc_snr(S_series[0]))
    
    # 初期値 S_0 の生成（各 sparsity ごとに生成）
    S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_0 = S_0 / norm(S_0)
    
    # --- 各手法のモデルのインスタンス化 ---
    tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, name="pc")
    tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, name="co")
    tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, name="sgd")
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, name="pp")
    
    # --- 各手法の実行関数 ---
    def run_tv_sem_pc() -> Tuple[List[np.ndarray], List[float]]:
        estimates, cost_values = tv_sem_pc.run(X)
        return estimates, cost_values

    def run_tv_sem_co() -> Tuple[List[np.ndarray], List[float]]:
        estimates, cost_values = tv_sem_co.run(X)
        return estimates, cost_values

    def run_tv_sem_sgd() -> Tuple[List[np.ndarray], List[float]]:
        estimates, cost_values = tv_sem_sgd.run(X)
        return estimates, cost_values

    def run_tv_sem_pp() -> List[np.ndarray]:
        estimates = tv_sem_pp.run(X)
        return estimates

    # --- 実行対象の関数リスト作成 ---
    job_list = []
    if run_pc_flag:
        job_list.append(delayed(run_tv_sem_pc)())
    if run_co_flag:
        job_list.append(delayed(run_tv_sem_co)())
    if run_sgd_flag:
        job_list.append(delayed(run_tv_sem_sgd)())
    if run_pp_flag:
        job_list.append(delayed(run_tv_sem_pp)())

    # 並列実行（n_jobs は適宜調整）
    results = Parallel(n_jobs=4)(job_list)

    # --- 各手法の結果から最終時刻のNSEを算出 ---
    idx_result: int = 0
    if run_pc_flag:
        estimates_pc, _ = results[idx_result]
        error_pc = (norm(estimates_pc[-1] - S_series[-1]) ** 2) / (norm(S_0 - S_series[-1]) ** 2)
        final_NSE_pc.append(error_pc)
        idx_result += 1
    else:
        final_NSE_pc.append(np.nan)

    if run_co_flag:
        estimates_co, _ = results[idx_result]
        error_co = (norm(estimates_co[-1] - S_series[-1]) ** 2) / (norm(S_0 - S_series[-1]) ** 2)
        final_NSE_co.append(error_co)
        idx_result += 1
    else:
        final_NSE_co.append(np.nan)

    if run_sgd_flag:
        estimates_sgd, _ = results[idx_result]
        error_sgd = (norm(estimates_sgd[-1] - S_series[-1]) ** 2) / (norm(S_0 - S_series[-1]) ** 2)
        final_NSE_sgd.append(error_sgd)
        idx_result += 1
    else:
        final_NSE_sgd.append(np.nan)

    if run_pp_flag:
        estimates_pp = results[idx_result]
        error_pp = (norm(estimates_pp[-1] - S_series[-1]) ** 2) / (norm(S_0 - S_series[-1]) ** 2)
        final_NSE_pp.append(error_pp)
        idx_result += 1
    else:
        final_NSE_pp.append(np.nan)

# --- 結果のプロット ---
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
plt.ylabel('Final NSE')
plt.grid(True, which='both')
plt.legend()

# 結果画像の保存ファイル名作成（パラメータ情報・タイムスタンプ付）
timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# __file__ が定義されていない場合は 'simulation.py' とする
notebook_filename: str = os.path.basename(__file__) if '__file__' in globals() else 'simulation.py'
filename: str = (
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

today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = os.path.join('.', 'result', today_str, 'images')
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

# --- コードファイルのバックアップ ---
copy_script_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
try:
    shutil.copy(notebook_filename, copy_script_path)
    print(f"Script file copied to: {copy_script_path}")
except Exception as e:
    print("Backup failed:", e)
