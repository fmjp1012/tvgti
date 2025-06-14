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
from scipy.linalg import inv, eigvals, norm
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager

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
# Proposed手法は、rを変化させるシミュレーションとqを変化させるシミュレーションに分割
run_pp_r_flag: bool = True    # rを変化させるシミュレーションの実行をONにする場合はTrue
run_pp_q_flag: bool = False   # qを変化させるシミュレーションの実行をONにする場合はTrue

#----------------------------------------------------
# パラメータの設定
N: int = 10
T: int = 400
sparsity: float = 0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True

seed: int = 30
np.random.seed(seed)

# TV-SEMシミュレーション
S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)

snr_before: float = calc_snr(S_series[0])
print("Before scaling, SNR =", snr_before)


# オンラインTV-SEMパラメータ
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

# オフライン解の計算（全データXを使用）
S_offline = solve_offline_sem(X, 0, True)
S_offline = (S_offline + S_offline.T) / 2
np.fill_diagonal(S_offline, 0)
offline_nse = norm(S_offline - S_series[-1])**2 / norm(S_0 - S_series[-1])**2
print("Offline NSE =", offline_nse)

# r と q の範囲
r_values = range(1, 4, 1)  # [1, 5, 15, ...] -> [1, 2, 3]に変更
q_values = range(1, 4, 1)  # [1, 5, 15, ...] -> [1, 2, 3]に変更

#----------------------------------------------------
# モデルのインスタンス化（PP手法以外）
tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, name="pc")
tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, name="co")
tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, name="sgd")

#----------------------------------------------------
# Proposed手法の実行関数の変更
# rho_val, mu_lambda_val を引数として受け取るように変更
def run_tv_sem_pp(r_val: int, q_val: int, rho_val: float, mu_lambda_val: float) -> List[np.ndarray]:
    """
    r_val, q_val, rho_val, mu_lambda_val を指定して Proposed (PP) 手法を走らせる。
    そのときの推定行列列 estimates_pp を返す。
    """
    tv_sem_pp_tmp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(
        N, S_0, r_val, q_val, rho_val, mu_lambda_val,
        name=f"pp_r{r_val}_q{q_val}_rho{rho_val}_mu{mu_lambda_val}"
    )
    estimates_pp_tmp = tv_sem_pp_tmp.run(X)
    return estimates_pp_tmp

#----------------------------------------------------
# Proposed手法: r を変化させる場合の並列実行
pp_estimates_for_r = {}  # rごとの推定列を保存
if run_pp_r_flag:
    # 並列実行のためのパラメータリスト作成
    params_r = [(r, q, rho, mu) for r, q, rho, mu in zip(r_values, q_values, [0.0312, 0.0707, 0.149], [0.0389, 0.116, 0.247])]
    results_r = Parallel(n_jobs=-1)(
        delayed(run_tv_sem_pp)(r, q, rho, mu) for r, q, rho, mu in params_r
    )
    pp_estimates_for_r = {r: result for r, result in zip(r_values, results_r)}

#----------------------------------------------------
# Proposed手法: q を変化させる場合の並列実行
pp_estimates_for_q = {}  # qごとの推定列を保存
if run_pp_q_flag:
    params_q = [(r, q, rho, mu) for r, q, rho, mu in zip(r_values, q_values, [0.015, 0.0297, 0.0321], [0.00857, 0.0305, 0.0264])]
    results_q = Parallel(n_jobs=4)(
        delayed(run_tv_sem_pp)(r, q, rho, mu) for r, q, rho, mu in params_q
    )
    pp_estimates_for_q = {q: result for q, result in zip(q_values, results_q)}

#----------------------------------------------------
# それぞれの推定結果について、NSE（正規化二乗誤差）を計算する関数
def calc_nse_series(estimates_list: List[np.ndarray], true_S_list: List[np.ndarray], S_0_ref: np.ndarray) -> List[float]:
    """
    推定行列リストと真の行列リストから、各時刻 t の
    NSE = ||S_est(t) - S_true(t)||^2 / ||S_0_ref - S_true(t)||^2 を計算して返す
    """
    nse_arr = []
    for i, estimate in enumerate(estimates_list):
        denom = norm(S_0_ref - true_S_list[i])**2
        numer = norm(estimate - true_S_list[i])**2 if i < len(true_S_list) else norm(estimate - true_S_list[-1])**2
        if denom < 1e-12:
            nse_val = 0.0
        else:
            nse_val = numer / denom
        nse_arr.append(nse_val)
    return nse_arr

#----------------------------------------------------
# r を変化させた Proposed 手法の NSE を計算
pp_error_for_r = {}  # {r_val: [NSE(t=0), NSE(t=1), ...], ...}
if run_pp_r_flag:
    for r_val in r_values:
        estimates_r = pp_estimates_for_r[r_val]
        pp_error_for_r[r_val] = calc_nse_series(estimates_r, S_series, S_0)

#----------------------------------------------------
# q を変化させた Proposed 手法の NSE を計算
pp_error_for_q = {}  # {q_val: [NSE(t=0), NSE(t=1), ...], ...}
if run_pp_q_flag:
    for q_val in q_values:
        estimates_q = pp_estimates_for_q[q_val]
        pp_error_for_q[q_val] = calc_nse_series(estimates_q, S_series, S_0)

timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)

#--------------------------
# Proposed手法: r を変化
#--------------------------
if run_pp_r_flag:
    plt.figure(figsize=(10, 6))
    for r_val in r_list:
        label_str = f'PP (r={r_val}, q={q_fixed})'
        plt.plot(pp_error_for_r[r_val], label=label_str)
    # オフライン解のNSEを横線で追加
    plt.axhline(y=offline_nse, color='k', linestyle='--', label='Offline solution')
    
    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('t')
    plt.ylabel('NSE')
    plt.grid(True, which="both")
    plt.legend()
    filename_r = f'timestamp{timestamp}_compare_PP_r_values.png'
    plt.savefig(os.path.join(save_path, filename_r))
    plt.show()

#--------------------------
# Proposed手法: q を変化
#--------------------------
if run_pp_q_flag:
    plt.figure(figsize=(10, 6))
    for q_val in q_list:
        label_str = f'PP (r={r_fixed}, q={q_val})'
        plt.plot(pp_error_for_q[q_val], label=label_str)
    # オフライン解のNSEを横線で追加
    plt.axhline(y=offline_nse, color='k', linestyle='--', label='Offline solution')
    
    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('t')
    plt.ylabel('NSE')
    plt.grid(True, which="both")
    plt.legend()
    filename_q = f'timestamp{timestamp}_compare_PP_q_values.png'
    plt.savefig(os.path.join(save_path, filename_q))
    plt.show()

#--------------------------
# スクリプトのバックアップコピー
#--------------------------
notebook_filename: str = os.path.basename(__file__)
copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")

# --------------------------
# 最終推定結果と再構成誤差の表示
# --------------------------
print("\n--- Final Estimates and Reconstruction Errors ---\n")

# オフライン解の表示
print("Offline solution:")

print(S_offline)
offline_error_val = norm(X - S_offline @ X)**2
print("Offline ||X - S*X||^2 =", offline_error_val)

# PC手法（Prediction Correction）の最終推定結果（run_pc_flagがTrueの場合）
if run_pc_flag:
    pc_estimates = tv_sem_pc.run(X)
    final_S_pc = pc_estimates[-1]
    error_pc = norm(X - final_S_pc @ X)**2
    print("\nPC method final estimate:")
    print(final_S_pc)
    print("PC ||X - S*X||^2 =", error_pc)

# CO手法（Correction Only）の最終推定結果（run_co_flagがTrueの場合）
if run_co_flag:
    co_estimates = tv_sem_co.run(X)
    final_S_co = co_estimates[-1]
    error_co = norm(X - final_S_co @ X)**2
    print("\nCO method final estimate:")
    print(final_S_co)
    print("CO ||X - S*X||^2 =", error_co)

# SGD手法の最終推定結果（run_sgd_flagがTrueの場合）
if run_sgd_flag:
    sgd_estimates = tv_sem_sgd.run(X)
    final_S_sgd = sgd_estimates[-1]
    error_sgd = norm(X - final_S_sgd @ X)**2
    print("\nSGD method final estimate:")
    print(final_S_sgd)
    print("SGD ||X - S*X||^2 =", error_sgd)

# Proposed手法: r を変化させる場合の最終推定結果（run_pp_r_flagがTrueの場合）
if run_pp_r_flag:
    for r_val in r_list:
        final_S_pp_r = pp_estimates_for_r[r_val][-1]
        error_pp_r = norm(X - final_S_pp_r @ X)**2
        print(f"\nProposed PP method (r={r_val}, q={q_fixed}) final estimate:")
        print(final_S_pp_r)
        print(f"PP (r={r_val}) ||X - S*X||^2 =", error_pp_r)

# Proposed手法: q を変化させる場合の最終推定結果（run_pp_q_flagがTrueの場合）
if run_pp_q_flag:
    for q_val in q_list:
        final_S_pp_q = pp_estimates_for_q[q_val][-1]
        error_pp_q = norm(X - final_S_pp_q @ X)**2
        print(f"\nProposed PP method (r={r_fixed}, q={q_val}) final estimate:")
        print(final_S_pp_q)
        print(f"PP (q={q_val}) ||X - S*X||^2 =", error_pp_q)
