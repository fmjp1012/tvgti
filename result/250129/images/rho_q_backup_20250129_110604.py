import shutil
import os
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

plt.rc('text',usetex=True)
plt.rc('font',family="serif")
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
run_pc_flag: bool = True     # Prediction Correction
run_co_flag: bool = True     # Correction Only
run_sgd_flag: bool = True    # SGD
run_pp_flag: bool = True     # Proposed
#----------------------------------------------------

# パラメータの設定
N: int = 10
T: int = 20000
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

# その他のパラメータ（Proposed手法関連）
r: int = 4  # window size
q: int = 10  # number of processors
rho: float = 0.15  # 試行回数の設定
mu_lambda: float = 1

# モデルのインスタンス化
tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, name="pc")
tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, name="co")
tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, name="sgd")
tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, name="pp")

def run_tv_sem_pp(r_val: int, q_val: int) -> List[np.ndarray]:
    """
    r_val, q_val を指定して Proposed (PP) 手法を走らせる。
    そのときの推定行列列 estimates_pp を返す。
    """
    tv_sem_pp_tmp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(
        N, S_0, r_val, q_val, rho, mu_lambda, name=f"pp_r{r_val}_q{q_val}"
    )
    estimates_pp_tmp = tv_sem_pp_tmp.run(X)
    return estimates_pp_tmp

#----------------------------------------------------
# ここで実行対象の関数だけリストを作って並列実行
job_list = []

results = Parallel(n_jobs=4)(job_list)
#----------------------------------------------------

# それぞれの結果を受け取る格納リスト
estimates_pc: List[np.ndarray] = []
cost_values_pc: List[float] = []
estimates_co: List[np.ndarray] = []
cost_values_co: List[float] = []
estimates_sgd: List[np.ndarray] = []
cost_values_sgd: List[float] = []

#====================================================
# ここからがポイント：pp手法で r または q を変化させて比較する
#====================================================

# 1) r を複数試す場合 (qは固定)
r_list = [2, 4, 8]  # 例としていくつかピックアップ
q_fixed = 10       # 固定する q の値
pp_estimates_for_r = {}  # rごとの推定列を保存

if run_pp_flag:
    for r_val in r_list:
        print(f"Run pp method with r={r_val}, q={q_fixed}")
        est_pp_r = run_tv_sem_pp(r_val, q_fixed)
        pp_estimates_for_r[r_val] = est_pp_r

# 2) q を複数試す場合 (rは固定)
q_list = [2, 10, 20]  # 例としていくつかピックアップ
r_fixed = 4          # 固定する r の値
pp_estimates_for_q = {}  # qごとの推定列を保存

if run_pp_flag:
    for q_val in q_list:
        print(f"Run pp method with r={r_fixed}, q={q_val}")
        est_pp_q = run_tv_sem_pp(r_fixed, q_val)
        pp_estimates_for_q[q_val] = est_pp_q

#----------------------------------------------------
# それぞれの推定結果について、NSE（正規化二乗誤差）を計算する関数
#----------------------------------------------------
def calc_nse_series(estimates_list: List[np.ndarray], true_S_list: List[np.ndarray], S_0_ref: np.ndarray) -> List[float]:
    """
    推定行列リストと真の行列リストから、各時刻 t の
    NSE = ||S_est(t) - S_true(t)||^2 / ||S_0_ref - S_true(t)||^2 を計算して返す
    """
    nse_arr = []
    for i, estimate in enumerate(estimates_list):
        # 分母は S_0_ref と真の S_series[i] の誤差ノルム
        denom = norm(S_0_ref - true_S_list[i])**2
        numer = norm(estimate - true_S_list[i])**2
        if denom < 1e-12:
            # 万が一分母が 0 に近い場合のガード
            nse_val = 0.0
        else:
            nse_val = numer / denom
        nse_arr.append(nse_val)
    return nse_arr

#----------------------------------------------------
# r を変化させた pp の NSE を計算
#----------------------------------------------------
pp_error_for_r = {}  # {r_val: [NSE(t=0), NSE(t=1), ...], ...}
if run_pp_flag:
    for r_val in r_list:
        estimates_r = pp_estimates_for_r[r_val]
        pp_error_for_r[r_val] = calc_nse_series(estimates_r, S_series, S_0)

#----------------------------------------------------
# q を変化させた pp の NSE を計算
#----------------------------------------------------
pp_error_for_q = {}  # {q_val: [NSE(t=0), NSE(t=1), ...], ...}
if run_pp_flag:
    for q_val in q_list:
        estimates_q = pp_estimates_for_q[q_val]
        pp_error_for_q[q_val] = calc_nse_series(estimates_q, S_series, S_0)

timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)

#--------------------------
# Proposed手法: r を変化
#--------------------------
plt.figure(figsize=(10,6))
for r_val in r_list:
    label_str = f'PP (r={r_val}, q={q_fixed})'
    plt.plot(pp_error_for_r[r_val], label=label_str)

plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('NSE')
plt.grid(True, "both")
plt.legend()
filename_r = f'compare_PP_r_values_{timestamp}.png'
plt.savefig(os.path.join(save_path, filename_r))
plt.show()


#--------------------------
# Proposed手法: q を変化
#--------------------------
plt.figure(figsize=(10,6))
for q_val in q_list:
    label_str = f'PP (r={r_fixed}, q={q_val})'
    plt.plot(pp_error_for_q[q_val], label=label_str)

plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('NSE')
plt.grid(True, "both")
plt.legend()
filename_q = f'compare_PP_q_values_{timestamp}.png'
plt.savefig(os.path.join(save_path, filename_q))
plt.show()


#--------------------------
# スクリプトのバックアップコピー
#--------------------------
notebook_filename: str = "rho_q.py"  # 元ファイル名（必要に応じて変更）
copy_ipynb_path: str = os.path.join(save_path, f"rho_q_backup_{timestamp}.py")
shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
