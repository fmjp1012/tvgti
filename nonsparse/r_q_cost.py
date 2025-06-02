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
N: int = 5  # 10 -> 5に変更
T: int = 100  # 40000 -> 100に変更
sparsity: float = 0
max_weight: float = 0.5
variance_e: float = 0.5
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

# --------------------------
# オフライン解の計算（全データ X を使用）
# --------------------------
S_offline = solve_offline_sem(X, 0, False)
S_offline = (S_offline + S_offline.T) / 2  # 対称化
np.fill_diagonal(S_offline, 0)            # 対角成分ゼロ化
offline_nse = norm(S_offline - S_series[-1])**2 / norm(S_0 - S_series[-1])**2 if (denom := norm(S_0 - S_series[-1])**2) > 1e-12 else 0.0
offline_cost = (1/(2*T)) * norm(X - S_offline @ X)**2
print("Offline NSE =", offline_nse)
print("Offline ||X - S*X||^2 =", offline_cost)

# --------------------------
# オフライン解の計算（部分データを用いる場合）
# --------------------------
n_splits = 3  # 10 -> 3に変更
offline_solutions_list = []
offline_nses_list = []
offline_costs_list = []
true_costs_list = []  # 追加：真の解によるコスト推移を格納

print("部分データを用いたオフライン解の評価:")
for i in range(1, n_splits + 1):
    T_partial = int(i * T / n_splits)
    # X_partial は全データの先頭 T_partial 個分を使用（X の形状が (N, T) であると仮定）
    X_partial = X[:, :T_partial]
    
    # オフライン解の計算（部分データ版）
    S_offline_partial = solve_offline_sem(X_partial, 0, False)
    S_offline_partial = (S_offline_partial + S_offline_partial.T) / 2  # 対称化
    np.fill_diagonal(S_offline_partial, 0)  # 対角成分ゼロ化
    offline_solutions_list.append(S_offline_partial)
    
    # 部分データにおけるオフライン解のコスト計算
    cost_partial = (1/(2*T_partial)) * norm(X_partial - S_offline_partial @ X_partial)**2
    offline_costs_list.append(cost_partial)
    
    # NSE の計算（真の解 S_series[-1] と比較）
    denom = norm(S_0 - S_series[-1])**2
    nse_partial = norm(S_offline_partial - S_series[-1])**2 / denom if denom > 1e-12 else 0.0
    offline_nses_list.append(nse_partial)
    
    # 追加：部分データにおける真の解（S_series[-1]）のコスト計算
    true_cost_partial = (1/(2*T_partial)) * norm(X_partial - S_series[-1] @ X_partial)**2
    true_costs_list.append(true_cost_partial)
    
    print(f"  データ数 {T_partial} 個: Offline NSE = {nse_partial:.4e}, Offline cost = {cost_partial:.4e}, True cost = {true_cost_partial:.4e}")

# 部分データにおけるOffline NSE, Costの推移をプロット（ここでは後で全体比較用に利用）
data_counts = [int(i * T / n_splits) for i in range(1, n_splits + 1)]

# --- rを変化させる場合 ---
r_list = [1, 2, 4]           # 試すrの値 (5個->3個に変更)
q_fixed = 1                         # qは固定
rho_list_r = [0.0312, 0.0707, 0.149]       # rごとのrhoの値（r_listと同じ長さ）
mu_lambda_list_r = [0.0389, 0.116, 0.247]  # rごとのmu_lambdaの値

# --- qを変化させる場合 ---
q_list = [1, 2, 4]            # 試すqの値 (5個->3個に変更)
r_fixed = 1                          # rは固定
rho_list_q = [0.015, 0.0297, 0.0321]      # qごとのrhoの値（q_listと同じ長さ）
mu_lambda_list_q = [0.00857, 0.0305, 0.0264]  # qごとのmu_lambdaの値

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
pp_estimates_for_r = {}  # rごとの推定結果を保存
if run_pp_r_flag:
    # 並列実行のためのパラメータリスト作成
    params_r = [(r, q_fixed, rho, mu) for r, rho, mu in zip(r_list, rho_list_r, mu_lambda_list_r)]
    results_r = Parallel(n_jobs=-1)(
        delayed(run_tv_sem_pp)(r, q, rho, mu) for r, q, rho, mu in params_r
    )
    pp_estimates_for_r = {r: result for r, result in zip(r_list, results_r)}

#----------------------------------------------------
# Proposed手法: q を変化させる場合の並列実行
pp_estimates_for_q = {}  # qごとの推定結果を保存
if run_pp_q_flag:
    params_q = [(r_fixed, q, rho, mu) for q, rho, mu in zip(q_list, rho_list_q, mu_lambda_list_q)]
    results_q = Parallel(n_jobs=4)(
        delayed(run_tv_sem_pp)(r, q, rho, mu) for r, q, rho, mu in params_q
    )
    pp_estimates_for_q = {q: result for q, result in zip(q_list, results_q)}

#----------------------------------------------------
# 各推定結果について、NSE（正規化二乗誤差）を計算する関数
def calc_nse_series(estimates_list: List[np.ndarray], true_S_list: List[np.ndarray], S_0_ref: np.ndarray) -> List[float]:
    """
    推定行列リストと真の行列リストから、各時刻 t の
    NSE = ||S_est(t) - S_true(t)||^2 / ||S_0_ref - S_true(t)||^2 を計算して返す
    """
    nse_arr = []
    for i, estimate in enumerate(estimates_list):
        denom = norm(S_0_ref - true_S_list[i])**2
        numer = norm(estimate - true_S_list[i])**2 if i < len(true_S_list) else norm(estimate - true_S_list[-1])**2
        nse_val = numer / denom if denom > 1e-12 else 0.0
        nse_arr.append(nse_val)
    return nse_arr

#----------------------------------------------------
# Proposed手法: r を変化させた Proposed 手法の NSE を計算
pp_error_for_r = {}  # {r_val: [NSE(t=0), NSE(t=1), ...], ...}
if run_pp_r_flag:
    for r_val in r_list:
        estimates_r = pp_estimates_for_r[r_val]
        nse_series_r = calc_nse_series(estimates_r, S_series, S_0)
        pp_error_for_r[r_val] = nse_series_r

# Proposed手法: q を変化させた Proposed 手法の NSE を計算
pp_error_for_q = {}  # {q_val: [NSE(t=0), NSE(t=1), ...], ...}
if run_pp_q_flag:
    for q_val in q_list:
        estimates_q = pp_estimates_for_q[q_val]
        nse_series_q = calc_nse_series(estimates_q, S_series, S_0)
        pp_error_for_q[q_val] = nse_series_q

#----------------------------------------------------
# コスト計算の関数
def calc_cost_series(estimates_list: List[np.ndarray], X: np.ndarray) -> List[float]:
    """
    推定行列リストを受け取って、各時刻 t における
    cost(t) = ||X[:, :t+1] - S_est(t) @ X[:, :t+1]||^2 / (2*(t+1)) を計算して返す。
    """
    def compute_cost(S_est):
        return (1 / (2 * X.shape[1])) * norm(X - S_est @ X) ** 2
    
    # 各推定結果に対してコストを計算
    cost_list = []
    for estimate in estimates_list:
        cost_val = compute_cost(estimate)
        cost_list.append(cost_val)
    return cost_list

# rを変化させた場合のコスト推移
pp_cost_for_r = {}
if run_pp_r_flag:
    def compute_cost_for_r(r_val):
        estimates_r = pp_estimates_for_r[r_val]
        cost_series_r = calc_cost_series(estimates_r, X)
        return cost_series_r

    for r_val in r_list:
        pp_cost_for_r[r_val] = compute_cost_for_r(r_val)

# qを変化させた場合のコスト推移
pp_cost_for_q = {}
if run_pp_q_flag:
    def compute_cost_for_q(q_val):
        estimates_q = pp_estimates_for_q[q_val]
        cost_series_q = calc_cost_series(estimates_q, X)
        return cost_series_q

    for q_val in q_list:
        pp_cost_for_q[q_val] = compute_cost_for_q(q_val)

#----------------------------------------------------
# 結果のプロット
# NSE推移（r変化）と最終NSE
if run_pp_r_flag:
    plt.figure(figsize=(12, 5))
    
    # NSE推移のプロット
    plt.subplot(1, 2, 1)
    for r_val in r_list:
        nse_series = pp_error_for_r[r_val]
        plt.plot(nse_series, label=f'r={r_val}')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('NSE')
    plt.title('NSE progression (varying r)')
    plt.legend()
    plt.grid(True, which='both')
    
    # 最終NSEのプロット
    plt.subplot(1, 2, 2)
    final_nse_r = [pp_error_for_r[r_val][-1] for r_val in r_list]
    plt.plot(r_list, final_nse_r, 'o-')
    plt.yscale('log')
    plt.xlabel('r')
    plt.ylabel('Final NSE')
    plt.title('Final NSE vs r')
    plt.grid(True, which='both')
    
    plt.tight_layout()
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    notebook_filename = os.path.basename(__file__)
    filename_r = (
        f'timestamp{timestamp}_'
        f'result_N{N}_'
        f'{notebook_filename}_'
        f'T{T}_'
        f'r_comparison.png'
    )
    today_str = datetime.datetime.now().strftime('%y%m%d')
    save_path = f'./result/{today_str}/images'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename_r))
    plt.show()

# コスト推移（r変化）
if run_pp_r_flag:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for r_val in r_list:
        cost_series = pp_cost_for_r[r_val]
        plt.plot(cost_series, label=f'r={r_val}')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('Cost')
    plt.title('Cost progression (varying r)')
    plt.legend()
    plt.grid(True, which='both')
    
    plt.subplot(1, 2, 2)
    final_cost_r = [pp_cost_for_r[r_val][-1] for r_val in r_list]
    plt.plot(r_list, final_cost_r, 'o-')
    plt.yscale('log')
    plt.xlabel('r')
    plt.ylabel('Final Cost')
    plt.title('Final Cost vs r')
    plt.grid(True, which='both')
    
    plt.tight_layout()
    
    filename_cost_r = (
        f'timestamp{timestamp}_'
        f'result_N{N}_'
        f'{notebook_filename}_'
        f'T{T}_'
        f'r_cost_comparison.png'
    )
    plt.savefig(os.path.join(save_path, filename_cost_r))
    plt.show()

# NSE推移（q変化）と最終NSE
if run_pp_q_flag:
    plt.figure(figsize=(12, 5))
    
    # NSE推移のプロット
    plt.subplot(1, 2, 1)
    for q_val in q_list:
        nse_series = pp_error_for_q[q_val]
        plt.plot(nse_series, label=f'q={q_val}')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('NSE')
    plt.title('NSE progression (varying q)')
    plt.legend()
    plt.grid(True, which='both')
    
    # 最終NSEのプロット
    plt.subplot(1, 2, 2)
    final_nse_q = [pp_error_for_q[q_val][-1] for q_val in q_list]
    plt.plot(q_list, final_nse_q, 'o-')
    plt.yscale('log')
    plt.xlabel('q')
    plt.ylabel('Final NSE')
    plt.title('Final NSE vs q')
    plt.grid(True, which='both')
    
    plt.tight_layout()
    
    filename_q = (
        f'timestamp{timestamp}_'
        f'result_N{N}_'
        f'{notebook_filename}_'
        f'T{T}_'
        f'q_comparison.png'
    )
    plt.savefig(os.path.join(save_path, filename_q))
    plt.show()

# Back up this script
copy_ipynb_path = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(__file__, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
