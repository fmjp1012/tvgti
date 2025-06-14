import shutil
import sys
import os
import datetime
from typing import List, Tuple, Dict

import numpy as np
from scipy.linalg import inv, eigvals, norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager

from utils import *
from models.tvgti_pc_nonsparse import TimeVaryingSEM as TimeVaryingSEM_PC_NONSPARSE
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED

# -----------------------------
# matplotlib の設定
# -----------------------------
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
run_pc_flag: bool = True     # Prediction Correction
run_co_flag: bool = True     # Correction Only
run_sgd_flag: bool = True    # SGD
run_pp_flag: bool = True     # Proposed
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

# ここで SNR ターゲットを指定 (例: 3.0)
snr_target: float = 2

seed: int = 30
np.random.seed(seed)

# ----------------------------------------------------
# ここで修正後の generate_piecewise_X_K を使い、
# snr_target を渡して S_series と X を生成
# ----------------------------------------------------
S_series, X = generate_piecewise_X_K_with_snr(
    N,
    T,
    S_is_symmetric,
    sparsity,
    max_weight,
    std_e,
    K,
    snr_target  # <-- ここが変更点
)

print("SNR check =", calc_snr(S_series[0]))

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

# その他のパラメータ
r: int = 4  # window size
q: int = 10  # number of processors
rho: float = 0.15
mu_lambda: float = 1

# モデルのインスタンス化
tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, name="pc")
tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, name="co")
tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, name="sgd")
tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, name="pp")

# 実行関数の定義
def run_tv_sem_pc():
    estimates_pc, cost_values_pc = tv_sem_pc.run(X)
    return estimates_pc, cost_values_pc

def run_tv_sem_co():
    estimates_co, cost_values_co = tv_sem_co.run(X)
    return estimates_co, cost_values_co

def run_tv_sem_sgd():
    estimates_sgd, cost_values_sgd = tv_sem_sgd.run(X)
    return estimates_sgd, cost_values_sgd

def run_tv_sem_pp():
    estimates_pp = tv_sem_pp.run(X)
    return estimates_pp

job_list = []
if run_pc_flag:
    job_list.append(delayed(run_tv_sem_pc)())
if run_co_flag:
    job_list.append(delayed(run_tv_sem_co)())
if run_sgd_flag:
    job_list.append(delayed(run_tv_sem_sgd)())
if run_pp_flag:
    job_list.append(delayed(run_tv_sem_pp)())

results = Parallel(n_jobs=4)(job_list)

estimates_pc = []
cost_values_pc = []
estimates_co = []
cost_values_co = []
estimates_sgd = []
cost_values_sgd = []
estimates_pp = []

idx_result = 0
if run_pc_flag:
    estimates_pc, cost_values_pc = results[idx_result]
    idx_result += 1
if run_co_flag:
    estimates_co, cost_values_co = results[idx_result]
    idx_result += 1
if run_sgd_flag:
    estimates_sgd, cost_values_sgd = results[idx_result]
    idx_result += 1
if run_pp_flag:
    estimates_pp = results[idx_result]
    idx_result += 1

# ここから結果の解析・可視化
S_opts: List[np.ndarray] = []
NSE_pc: List[float] = []
NSE_co: List[float] = []
NSE_sgd: List[float] = []
NSE_pp: List[float] = []
error_pc: List[float] = []
error_co: List[float] = []
error_sgd: List[float] = []
error_pp: List[float] = []

sum_error_pc: List[float] = []
sum_error_co: List[float] = []
sum_error_sgd: List[float] = []
sum_error_pp: List[float] = []

# 各時刻 t (もしくは各セグメント i) に対して真値 S_series[i] と比較
if run_pc_flag:
    for i, estimate in enumerate(estimates_pc):
        error_val = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_pc.append(error_val)
        sum_error_pc.append((estimate - S_series[i]).sum())

if run_co_flag:
    for i, estimate in enumerate(estimates_co):
        error_val = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_co.append(error_val)
        sum_error_co.append((estimate - S_series[i]).sum())

if run_sgd_flag:
    for i, estimate in enumerate(estimates_sgd):
        error_val = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_sgd.append(error_val)
        sum_error_sgd.append((estimate - S_series[i]).sum())

if run_pp_flag:
    for i, estimate in enumerate(estimates_pp):
        error_val = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_pp.append(error_val)
        sum_error_pp.append((estimate - S_series[i]).sum())

plt.figure(figsize=(10,6))

if run_co_flag:
    plt.plot(error_co, color='blue', label='Correction Only')
if run_pc_flag:
    plt.plot(error_pc, color='limegreen', label='Prediction Correction')
if run_sgd_flag:
    plt.plot(error_sgd, color='cyan', label='SGD')
if run_pp_flag:
    plt.plot(error_pp, color='red', label='Proposed')

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
    f'mu_lambda{mu_lambda}.png'
)

print(filename)
today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

# Back up this script
copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(__file__, copy_ipynb_path)  # notebook_filename -> __file__に変更
print(f"Notebook file copied to: {copy_ipynb_path}")
