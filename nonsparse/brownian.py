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
from joblib import Parallel, delayed, Memory
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

# ----------------------------------------------------
# Execution flags
run_pc_flag: bool = True     # Prediction Correction
run_co_flag: bool = True     # Correction Only
run_sgd_flag: bool = True    # SGD
run_pp_flag: bool = True     # Proposed
# ----------------------------------------------------

# Parameters
N: int = 5
T: int = 100
sparsity: float = 0 # 0 ~ 1
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True
std_S: float = 0.2
seed: int = 30
np.random.seed(seed)

# Generate piecewise data
S_series, X = generate_brownian_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K, std_S)
snr_before: float = calc_snr(S_series[0])
print("Before scaling, SNR =", snr_before)

# Online TV-SEM parameters
P: int = 1
C: int = 1
gamma: float = 0.999
alpha: float = 0.015
beta_pc: float = 0.015
beta_co: float = 0.02
beta_sgd: float = 0.02

# Initial guess
S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
S_0 = S_0 / norm(S_0)

# Proposed model parameters
r: int = 4  # window size
q: int = 10  # number of processors
rho: float = 0.15
mu_lambda: float = 1

# Instantiate models
tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, name="pc")
tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, name="co")
tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, name="sgd")
tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, name="pp")

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

# -----------------------------------------------------------------
# Run selected methods in parallel
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

# Unpack results
estimates_pc, cost_values_pc = [], []
estimates_co, cost_values_co = [], []
estimates_sgd, cost_values_sgd = [], []
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

# 並列に使うコア数(スレッド数)を指定: 
# -1 とすると環境の全コア数に依存して実行する
n_jobs = -1

lambda_reg = 0

offline_sol = solve_offline_sem(X, lambda_reg)

# -----------------------------------------------------------------
# 2) Compute the Normalized Squared Error = || S_est - S^*_t ||^2 / || S^*_t ||^2
# -----------------------------------------------------------------
NSE_pc: List[float] = []
NSE_co: List[float] = []
NSE_sgd: List[float] = []
NSE_pp: List[float] = []

# PC
if run_pc_flag:
    for i, estimate in enumerate(estimates_pc):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        NSE_pc.append(error_val)

# Correction Only
if run_co_flag:
    for i, estimate in enumerate(estimates_co):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        NSE_co.append(error_val)

# SGD
if run_sgd_flag:
    for i, estimate in enumerate(estimates_sgd):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        NSE_sgd.append(error_val)

# Proposed
if run_pp_flag:
    for i, estimate in enumerate(estimates_pp):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        NSE_pp.append(error_val)

# -----------------------------------------------------------------
# Plot
# -----------------------------------------------------------------
plt.figure(figsize=(10,6))

if run_co_flag:
    plt.plot(NSE_co, label='Correction Only', color='blue')
if run_pc_flag:
    plt.plot(NSE_pc, label='Prediction Correction', color='limegreen')
if run_sgd_flag:
    plt.plot(NSE_sgd, label='SGD', color='cyan')
if run_pp_flag:
    plt.plot(NSE_pp, label='Proposed', color='red')

plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('NSE')
plt.grid(True, which="both")
plt.legend()

timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
notebook_filename: str = os.path.basename(__file__)
filename: str = (
    f'timestamp{timestamp}_result_N{N}_notebook_filename{notebook_filename}_T{T}_maxweight{max_weight}_variancee{variance_e}_K{K}_'
    f'Sissymmetric{S_is_symmetric}_seed{seed}_P{P}_C{C}_gammma{gamma}_'
    f'alpha{alpha}_betapc{beta_pc}_betaco{beta_co}_betasgd{beta_sgd}_'
    f'r{r}_q{q}_rho{rho}_mulambda{mu_lambda}_std_S{std_S}.png'
)
today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

# Back up this script
copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(__file__, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
