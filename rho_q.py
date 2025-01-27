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

def solve_offline_sem(X_up_to_t: np.ndarray, lambda_reg: float) -> np.ndarray:
    N, t = X_up_to_t.shape
    S = cp.Variable((N, N), symmetric=True)
    
    # objective = (1/(2*t)) * cp.norm(X_up_to_t - S @ X_up_to_t, 'fro') + lambda_reg * cp.norm1(S)
    objective = (1/(2*t)) * cp.norm(X_up_to_t - S @ X_up_to_t, 'fro')
    
    constraints = [cp.diag(S) == 0]
    
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    prob.solve(solver=cp.SCS, verbose=False)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("CVXPY did not find an optimal solution.")
    
    S_opt = S.value
    return S_opt

def calc_snr(S: np.ndarray) -> float:
    """
    与えられた行列 S (NxN) に対して、
    SNR = (1/N) * tr( (I - S)^-1 * (I - S)^-T ) を計算する。
    """
    N = S.shape[0]
    I = np.eye(N)
    inv_mat = np.linalg.inv(I - S)  # (I - S)^-1
    # (I - S)^-1 (I - S)^-T = inv_mat @ inv_mat.T
    val = np.trace(inv_mat @ inv_mat.T)
    return val / N

def scale_S_for_target_snr(S: np.ndarray, snr_target: float,
                           tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """
    与えられた S (NxN) をスケーリングする係数 alpha を見つけて、
    (I - alpha*S) が可逆 & スペクトル半径 < 1 となる範囲で
    目標とする snr_target に近い SNR を実現するように返す。
    """
    # 前提: S は正方行列
    N = S.shape[0]
    
    # スペクトル半径を計算
    eigvals = np.linalg.eigvals(S)
    rho_S = max(abs(eigvals))
    
    # もし rho_S == 0 なら、S=0 の場合などで SNR=1 が常に得られる
    # ここでは簡単に場合分け
    if rho_S == 0:
        current_snr = calc_snr(S * 0.0)  # = 1/N * tr(I * I^T) = 1
        if abs(current_snr - snr_target) < tol:
            return S  # そのまま
        else:
            # どうにもならないので、とりあえず返しておく
            return S
    
    # alpha の上限: ここでは 1/(rho_S + ちょっとのマージン) とする
    alpha_high = 1.0 / rho_S * 0.999  # 安全のため少しだけ小さめにする
    alpha_low = 0.0
    
    # 2分探索
    for _ in range(max_iter):
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        
        # (I - alpha*S) が可逆かチェック -> np.linalg.inv がエラーを吐かないか確かめる
        try:
            tmp_snr = calc_snr(alpha_mid * S)
        except np.linalg.LinAlgError:
            # 可逆でなかったら、もう少し alpha を小さくする
            alpha_high = alpha_mid
            continue
        
        if tmp_snr > snr_target:
            # 目標より SNR が高いので、alpha を小さく
            alpha_high = alpha_mid
        else:
            # 目標より SNR が低いので、alpha を大きく
            alpha_low = alpha_mid
        
        # 収束チェック
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
notebook_filename: str = "rho_q.py"

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

copy_ipynb_path: str = os.path.join(save_path, f"rho_q_backup_{timestamp}.py")

shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
