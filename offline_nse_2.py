import shutil
import os
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

def solve_offline_sem(X_up_to_t: np.ndarray, lambda_reg: float = 0.0) -> np.ndarray:
    """
    Solve the offline SEM problem (at time t) on data X_up_to_t.
    Minimizes (1/2t)*||X - S X||_F^2 subject to diag(S) = 0.
    Optionally include an L1 term if needed (lambda_reg>0).
    """
    N, t = X_up_to_t.shape
    S = cp.Variable((N, N), symmetric=True)
    
    # For L1 regularization, uncomment:
    # objective = (1/(2*t)) * cp.norm(X_up_to_t - S @ X_up_to_t, 'fro') + lambda_reg * cp.norm(S, 1)
    objective = (1/(2*t)) * cp.norm(X_up_to_t - S @ X_up_to_t, 'fro')
    constraints = [cp.diag(S) == 0]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("CVXPY did not find an optimal solution at time t=%d." % t)
    
    return S.value

def calc_snr(S: np.ndarray) -> float:
    N = S.shape[0]
    I = np.eye(N)
    inv_mat = np.linalg.inv(I - S)
    val = np.trace(inv_mat @ inv_mat.T)
    return val / N

# Dummy scaling function for SNR (not essential for the NSE calculation)
def scale_S_for_target_snr(S: np.ndarray, snr_target: float,
                           tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    eigvals_S = np.linalg.eigvals(S)
    rho_S = max(abs(eigvals_S))
    if rho_S == 0:
        # If S=0, SNR=1 always
        return S
    alpha_high = 1.0 / rho_S * 0.999
    alpha_low = 0.0
    for _ in range(max_iter):
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        try:
            tmp_snr = calc_snr(alpha_mid * S)
        except np.linalg.LinAlgError:
            alpha_high = alpha_mid
            continue
        if tmp_snr > snr_target:
            alpha_high = alpha_mid
        else:
            alpha_low = alpha_mid
        if abs(tmp_snr - snr_target) < tol:
            break
    alpha_star = 0.5 * (alpha_low + alpha_high)
    return alpha_star * S

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
N: int = 10
T: int = 20000
sparsity: float = 0 # 0 ~ 1
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True
seed: int = 30
np.random.seed(seed)

# Generate piecewise data
S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)
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

# For PC
if run_pc_flag:
    for i, est in enumerate(estimates_pc):
        nse_val = norm(est - offline_sol)**2 / (norm(offline_sol)**2 + 1e-12)
        NSE_pc.append(nse_val)

# For CO
if run_co_flag:
    for i, est in enumerate(estimates_co):
        nse_val = norm(est - offline_sol)**2 / (norm(offline_sol)**2 + 1e-12)
        NSE_co.append(nse_val)

# For SGD
if run_sgd_flag:
    for i, est in enumerate(estimates_sgd):
        nse_val = norm(est - offline_sol)**2 / (norm(offline_sol)**2 + 1e-12)
        NSE_sgd.append(nse_val)

# For Proposed
if run_pp_flag:
    for i, est in enumerate(estimates_pp):
        nse_val = norm(est - offline_sol)**2 / (norm(offline_sol)**2 + 1e-12)
        NSE_pp.append(nse_val)

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
filename: str = (
    f'result_N{N}_T{T}_maxweight{max_weight}_variancee{variance_e}_K{K}_'
    f'Sissymmetric{S_is_symmetric}_seed{seed}_P{P}_C{C}_gammma{gamma}_'
    f'alpha{alpha}_betapc{beta_pc}_betaco{beta_co}_betasgd{beta_sgd}_'
    f'r{r}_q{q}_rho{rho}_mulambda{mu_lambda}_timestamp{timestamp}.png'
)
today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

# Back up this script
notebook_filename: str = "offline_nse.py"
copy_ipynb_path: str = os.path.join(save_path, f"offline_nse_2_backup_{timestamp}.py")
shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
