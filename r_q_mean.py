import shutil
import os
import datetime
from typing import List, Dict

import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from utils import *
from models.tvgti_pc_nonsparse import TimeVaryingSEM as TimeVaryingSEM_PC_NONSPARSE
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED

#--------------------------
# プロット等の設定
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

#--------------------------
# メソッド実行のフラグ
run_pc_flag: bool = False     # Prediction Correction
run_co_flag: bool = False     # Correction Only
run_sgd_flag: bool = False    # SGD
# Proposed手法について、rまたはqを変化させるシミュレーション
run_pp_r_flag: bool = True    # rを変化させるシミュレーションを有効にする
run_pp_q_flag: bool = False   # qを変化させるシミュレーションを有効にする

#--------------------------
# パラメータ設定
N: int = 10
T: int = 40000
sparsity: float = 0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True

# TV-SEMのオンラインパラメータ
P: int = 1
C: int = 1
gamma: float = 0.999
alpha: float = 0.015
beta_pc: float = 0.015
beta_co: float = 0.02
beta_sgd: float = 0.02

# Proposed手法のパラメータ（rを変化させる場合）
r_list = [1, 2, 4, 8, 40]             # 試すrの値
q_fixed = 1                           # qは固定
rho_list_r = [0.0312, 0.0707, 0.134, 0.237, 1.034]       # rごとのrhoの値（r_listと同じ長さ）
mu_lambda_list_r = [0.0389, 0.116, 0.0867, 0.041, 0.0154]  # rごとのmu_lambdaの値

# Proposed手法（qを変化させる場合）のパラメータ
q_list = [1, 2, 4, 8, 40]
r_fixed = 1
rho_list_q = [0.0312, 0.0297, 0.0321, 0.0281, 0.069]
mu_lambda_list_q = [0.0389, 0.0305, 0.0264, 0.00939, 1.202]

# 試行回数の設定
num_trials = 100
base_seed = 30

#--------------------------
# NSE（正規化二乗誤差）を計算する関数
def calc_nse_series(estimates_list: List[np.ndarray],
                    true_S_list: List[np.ndarray],
                    S_0_ref: np.ndarray) -> List[float]:
    nse_arr = []
    for i, estimate in enumerate(estimates_list):
        denom = norm(S_0_ref - true_S_list[i])**2
        numer = norm(estimate - true_S_list[i])**2 if i < len(true_S_list) else norm(estimate - true_S_list[-1])**2
        nse_val = 0.0 if denom < 1e-12 else numer / denom
        nse_arr.append(nse_val)
    return nse_arr

#--------------------------
# Proposed手法の実行関数（グローバル変数 S_0, X を使用）
def run_tv_sem_pp(r_val: int, q_val: int, rho_val: float, mu_lambda_val: float) -> List[np.ndarray]:
    tv_sem_pp_tmp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(
        N, S_0, r_val, q_val, rho_val, mu_lambda_val,
        name=f"pp_r{r_val}_q{q_val}_rho{rho_val}_mu{mu_lambda_val}"
    )
    estimates_pp_tmp = tv_sem_pp_tmp.run(X)
    return estimates_pp_tmp

#--------------------------
# 1回の試行を実行する関数
def run_trial(trial_seed: int) -> Dict:
    np.random.seed(trial_seed)
    # 各試行ごとにデータを生成
    S_series, X_local = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)
    S_0_local = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_0_local = S_0_local / norm(S_0_local)
    
    # run_tv_sem_pp関数で利用するため、グローバル変数に設定
    global S_0, X
    S_0 = S_0_local
    X = X_local
    
    errors = {}
    
    # Proposed手法：rを変化させる場合
    if run_pp_r_flag:
        errors['pp_r'] = {}
        for r_val, rho, mu in zip(r_list, rho_list_r, mu_lambda_list_r):
            estimates_pp = run_tv_sem_pp(r_val, q_fixed, rho, mu)
            error_pp = calc_nse_series(estimates_pp, S_series, S_0_local)
            errors['pp_r'][r_val] = error_pp
            
    # Proposed手法：qを変化させる場合
    if run_pp_q_flag:
        errors['pp_q'] = {}
        for q_val, rho, mu in zip(q_list, rho_list_q, mu_lambda_list_q):
            estimates_pp = run_tv_sem_pp(r_fixed, q_val, rho, mu)
            error_pp = calc_nse_series(estimates_pp, S_series, S_0_local)
            errors['pp_q'][q_val] = error_pp

    return errors

#--------------------------
# 並列処理で各試行を実行
trial_seeds = [base_seed + i for i in range(num_trials)]
results = Parallel(n_jobs=-1)(
    delayed(run_trial)(seed) for seed in trial_seeds
)

#--------------------------
# 各パラメータごとにエラーを平均する
avg_errors_pp_r = {}
if run_pp_r_flag:
    for r_val in r_list:
        sum_error = np.zeros(T)
        for trial_result in results:
            sum_error += np.array(trial_result['pp_r'][r_val])
        avg_errors_pp_r[r_val] = sum_error / num_trials

avg_errors_pp_q = {}
if run_pp_q_flag:
    for q_val in q_list:
        sum_error = np.zeros(T)
        for trial_result in results:
            sum_error += np.array(trial_result['pp_q'][q_val])
        avg_errors_pp_q[q_val] = sum_error / num_trials

#--------------------------
# 結果のプロットと保存
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
today_str = datetime.datetime.now().strftime('%y%m%d')
save_path = os.path.join('./result', today_str, 'images')
os.makedirs(save_path, exist_ok=True)

if run_pp_r_flag:
    plt.figure(figsize=(10, 6))
    for r_val in r_list:
        label_str = f'PP (r={r_val}, q={q_fixed})'
        plt.plot(avg_errors_pp_r[r_val], label=label_str)
    plt.yscale('log')
    plt.xlim(0, T)
    plt.xlabel('t')
    plt.ylabel('Average NSE')
    plt.grid(True, which="both")
    plt.legend()
    filename_r = f'timestamp{timestamp}_compare_PP_r_values_avg.png'
    plt.savefig(os.path.join(save_path, filename_r))
    plt.show()

if run_pp_q_flag:
    plt.figure(figsize=(10, 6))
    for q_val in q_list:
        label_str = f'PP (r={r_fixed}, q={q_val})'
        plt.plot(avg_errors_pp_q[q_val], label=label_str)
    plt.yscale('log')
    plt.xlim(0, T)
    plt.xlabel('t')
    plt.ylabel('Average NSE')
    plt.grid(True, which="both")
    plt.legend()
    filename_q = f'timestamp{timestamp}_compare_PP_q_values_avg.png'
    plt.savefig(os.path.join(save_path, filename_q))
    plt.show()

#--------------------------
# スクリプトのバックアップコピー
notebook_filename = os.path.basename(__file__)
copy_ipynb_path = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
