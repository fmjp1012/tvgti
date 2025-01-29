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

plt.rc('text',usetex=True)
plt.rc('font',family="serif")
plt.rcParams["font.family"] = "Times New Roman"      #全体のフォントを設定
plt.rcParams["xtick.direction"] = "in"               #x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams["ytick.direction"] = "in"               #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams["xtick.minor.visible"] = True          #x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True          #y軸補助目盛りの追加
plt.rcParams["xtick.major.width"] = 1.5              #x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1.5              #y軸主目盛り線の線幅
plt.rcParams["xtick.minor.width"] = 1.0              #x軸補助目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 1.0              #y軸補助目盛り線の線幅
plt.rcParams["xtick.major.size"] = 10                #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 10                #y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 5                #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5                #y軸補助目盛り線の長さ
plt.rcParams["font.size"] = 15                       #フォントの大きさ

#----------------------------------------------------
# メソッドごとの実行スイッチ
run_pc_flag: bool = False     # Prediction Correction
run_co_flag: bool = False     # Correction Only
run_sgd_flag: bool = False    # SGD
run_pp_flag: bool = True     # Proposed
#----------------------------------------------------

# パラメータの設定
N: int = 30
T: int = 3000
sparsity: float = 0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 4
S_is_symmetric: bool = True

seed: int = 3
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

# その他のパラメータ
r: int = 4  # window size
q: int = 10  # number of processors
rho: float = 0.15  # 試行回数の設定
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

#----------------------------------------------------
# ここで実行対象の関数だけリストを作る
# (関数と手法名をまとめて管理するとあとで集計しやすい)
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
#----------------------------------------------------

# それぞれの結果を受け取る格納リスト
# （実行しないメソッドは空のままにする）
estimates_pc: List[np.ndarray] = []
cost_values_pc: List[float] = []
estimates_co: List[np.ndarray] = []
cost_values_co: List[float] = []
estimates_sgd: List[np.ndarray] = []
cost_values_sgd: List[float] = []
estimates_pp: List[np.ndarray] = []

# 実行した順番に応じて results から取り出す
idx_result: int = 0

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
    estimates_pp = results[idx_result] if run_pp_flag else []
    # すでに idx_result が並列実行数に達したらこの後は不要
    # （ただし if run_pp_flag: の条件内なので安全）
    idx_result += 1

# ここから結果の解析・可視化
# （実行したメソッドだけ処理をする）
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

# PC
if run_pc_flag:
    for i, estimate in enumerate(estimates_pc):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_pc.append(error_val)
        sum_error_pc.append((estimate - S_series[i]).sum())

# Correction Only
if run_co_flag:
    for i, estimate in enumerate(estimates_co):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_co.append(error_val)
        sum_error_co.append((estimate - S_series[i]).sum())

# SGD
if run_sgd_flag:
    for i, estimate in enumerate(estimates_sgd):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_sgd.append(error_val)
        sum_error_sgd.append((estimate - S_series[i]).sum())

# Proposed
if run_pp_flag:
    for i, estimate in enumerate(estimates_pp):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_pp.append(error_val)
        sum_error_pp.append((estimate - S_series[i]).sum())

# 結果のプロット
plt.figure(figsize=(10,6))

# Correction Only
if run_co_flag:
    plt.plot(error_co, color='blue', label='Correction Only')
# Prediction Correction
if run_pc_flag:
    plt.plot(error_pc, color='limegreen', label='Prediction Correction')
# SGD
if run_sgd_flag:
    plt.plot(error_sgd, color='cyan', label='SGD')
# Proposed
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
    f'mu_lambda{mu_lambda}_'
    f'timestamp{timestamp}.png'
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
