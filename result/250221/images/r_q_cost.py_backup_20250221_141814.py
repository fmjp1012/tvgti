import shutil
import os
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
N: int = 10
T: int = 40000
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

# --- rを変化させる場合 ---
r_list = [1, 2, 4, 8, 40]           # 試すrの値
q_fixed = 1                         # qは固定
rho_list_r = [0.0312, 0.0793, 0.134, 0.237, 1.034]       # rごとのrhoの値（r_listと同じ長さ）
mu_lambda_list_r = [0.0389, 0.33, 0.0867, 0.041, 0.0154]  # rごとのmu_lambdaの値

# --- qを変化させる場合 ---
q_list = [1, 2, 4, 8, 40]            # 試すqの値
r_fixed = 1                          # rは固定
rho_list_q = [0.015, 0.0297, 0.0321, 0.0281, 0.069]      # qごとのrhoの値（q_listと同じ長さ）
mu_lambda_list_q = [0.00857, 0.0305, 0.0264, 0.00939, 1.202]  # qごとのmu_lambdaの値

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
# 並列化された calc_cost_series
def calc_cost_series(estimates_list: List[np.ndarray], X: np.ndarray) -> List[float]:
    """
    各推定行列に対してコスト計算を行う関数の並列実行版。
    Cost = ||X - S_est @ X||^2
    """
    def compute_cost(S_est):
        return np.linalg.norm(X - S_est @ X)**2

    cost_arr = Parallel(n_jobs=-1)(
        delayed(compute_cost)(S_est) for S_est in tqdm(estimates_list, desc="Calculating cost series", leave=False)
    )
    return cost_arr

#----------------------------------------------------
# Proposed手法: r を変化させた Proposed 手法の NSE を計算
pp_error_for_r = {}  # {r_val: [NSE(t=0), NSE(t=1), ...], ...}
if run_pp_r_flag:
    for r_val in r_list:
        estimates_r = pp_estimates_for_r[r_val]
        pp_error_for_r[r_val] = calc_nse_series(estimates_r, S_series, S_0)

#----------------------------------------------------
# Proposed手法: q を変化させた Proposed 手法の NSE を計算
pp_error_for_q = {}  # {q_val: [NSE(t=0), NSE(t=1), ...], ...}
if run_pp_q_flag:
    for q_val in q_list:
        estimates_q = pp_estimates_for_q[q_val]
        pp_error_for_q[q_val] = calc_nse_series(estimates_q, S_series, S_0)

#----------------------------------------------------
# Proposed手法: r を変化させた Proposed 手法のコスト関数の推移を
# 並列化して計算する
pp_cost_for_r = {}  # {r_val: [cost(t=0), cost(t=1), ...], ...}
if run_pp_r_flag:
    def compute_cost_for_r(r_val):
        estimates_r = pp_estimates_for_r[r_val]
        cost_series = calc_cost_series(estimates_r, X)
        return (r_val, cost_series)
    results_cost_r = Parallel(n_jobs=-1)(
        delayed(compute_cost_for_r)(r_val) for r_val in r_list
    )
    pp_cost_for_r = {r_val: cost_series for r_val, cost_series in results_cost_r}

timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)

#--------------------------
# Proposed手法: r を変化（NSEのプロット）
#--------------------------
if run_pp_r_flag:
    plt.figure(figsize=(10, 6))
    for r_val in r_list:
        label_str = f'PP (r={r_val}, q={q_fixed})'
        plt.plot(pp_error_for_r[r_val], label=label_str)

    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('Iteration (t)')
    plt.ylabel('NSE')
    plt.grid(True, which="both")
    plt.legend()
    filename_r = f'compare_PP_r_values_{timestamp}.png'
    plt.savefig(os.path.join(save_path, filename_r))
    plt.show()

#--------------------------
# Proposed手法: q を変化（NSEのプロット）
#--------------------------
if run_pp_q_flag:
    plt.figure(figsize=(10, 6))
    for q_val in q_list:
        label_str = f'PP (r={r_fixed}, q={q_val})'
        plt.plot(pp_error_for_q[q_val], label=label_str)

    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('Iteration (t)')
    plt.ylabel('NSE')
    plt.grid(True, which="both")
    plt.legend()
    filename_q = f'compare_PP_q_values_{timestamp}.png'
    plt.savefig(os.path.join(save_path, filename_q))
    plt.show()

#--------------------------
# Proposed手法: r を変化（コスト関数のプロット）
#--------------------------
if run_pp_r_flag:
    plt.figure(figsize=(10, 6))
    for r_val in r_list:
        label_str = f'PP (r={r_val}, q={q_fixed})'
        plt.plot(pp_cost_for_r[r_val], label=label_str)

    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('Iteration')
    plt.ylabel(r'Cost $\|X - S\,X\|$')
    plt.grid(True, which="both")
    plt.legend()
    filename_cost = f'cost_function_PP_r_{timestamp}.png'
    plt.savefig(os.path.join(save_path, filename_cost))
    plt.show()

#--------------------------
# スクリプトのバックアップコピー
#--------------------------
notebook_filename: str = os.path.basename(__file__)
copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
