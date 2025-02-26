import datetime
import os
import shutil
import sys
from typing import Dict, List, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import Manager
from scipy.linalg import eigvals, inv, norm
from tqdm import tqdm

from utils import *
from models.tvgti_pc_nonsparse import TimeVaryingSEM as TimeVaryingSEM_PC_NONSPARSE
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED

# Matplotlib configuration
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
N: int = 10
T: int = 10000
sparsity: float = 0.0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
S_is_symmetric: bool = True

seed: int = 30
np.random.seed(seed)

#----------------------------------------------------
# 変更ポイント: generate_linear_X を用いてシミュレーションデータを生成
#----------------------------------------------------
S_series, X = generate_linear_X(
    N=N,
    T=T,
    S_is_symmetric=S_is_symmetric,
    sparsity=sparsity,
    max_weight=max_weight,
    std_e=std_e
)

# 参考として SNR を計算
snr_before: float = calc_snr(S_series[0])
print("SNR at t=0 (S_series[0]):", snr_before)

# オンラインTV-SEMパラメータ
P: int = 1
C: int = 1
gamma: float = 0.999
alpha: float = 0.015
beta_pc: float = 0.015
beta_co: float = 0.015
beta_sgd: float = 0.015

# 初期値の設定 (推定開始時の行列)
S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
S_0 = S_0 / norm(S_0)  # とりあえず正則化

# その他のパラメータ
r: int = 4   # window size
q: int = 10  # number of processors
rho: float = 0.15
mu_lambda: float = 1

# モデルのインスタンス化
tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, name="pc")
tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, name="co")
tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, name="sgd")
tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, name="pp")

# 実行関数の定義
def run_tv_sem_pc() -> Tuple[List[np.ndarray], List[float]]:
    estimates_pc, cost_values_pc = tv_sem_pc.run(X)
    return estimates_pc, cost_values_pc

def run_tv_sem_co() -> Tuple[List[np.ndarray], List[float]]:
    estimates_co, cost_values_co = tv_sem_co.run(X)
    return estimates_co, cost_values_co

def run_tv_sem_sgd() -> Tuple[List[np.ndarray], List[float]]:
    estimates_sgd, cost_values_sgd = tv_sem_sgd.run(X)
    return estimates_sgd, cost_values_sgd

def run_tv_sem_pp() -> List[np.ndarray]:
    estimates_pp = tv_sem_pp.run(X)
    return estimates_pp

# 実行
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

# それぞれの結果を受け取る格納リスト
estimates_pc: List[np.ndarray] = []
cost_values_pc: List[float] = []
estimates_co: List[np.ndarray] = []
cost_values_co: List[float] = []
estimates_sgd: List[np.ndarray] = []
cost_values_sgd: List[float] = []
estimates_pp: List[np.ndarray] = []

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

# 推定誤差などの計算 (NSE など)
error_pc:  List[float] = []
error_co:  List[float] = []
error_sgd: List[float] = []
error_pp:  List[float] = []

for i in range(T):
    S_true = S_series[i]
    if run_pc_flag:
        err_val = (norm(estimates_pc[i] - S_true) ** 2) / (norm(S_0 - S_true) ** 2)
        error_pc.append(err_val)
    if run_co_flag:
        err_val = (norm(estimates_co[i] - S_true) ** 2) / (norm(S_0 - S_true) ** 2)
        error_co.append(err_val)
    if run_sgd_flag:
        err_val = (norm(estimates_sgd[i] - S_true) ** 2) / (norm(S_0 - S_true) ** 2)
        error_sgd.append(err_val)
    if run_pp_flag:
        err_val = (norm(estimates_pp[i] - S_true) ** 2) / (norm(S_0 - S_true) ** 2)
        error_pp.append(err_val)

# 結果のプロット
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

today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
