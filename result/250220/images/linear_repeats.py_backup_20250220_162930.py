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

# Matplotlib の設定
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

# ------------------------------
# シミュレーションの総実行回数
num_repeats: int = 5  # ここで繰り返し回数を指定

# 基本パラメータの設定
N: int = 10
T: int = 10000
sparsity: float = 0.0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
S_is_symmetric: bool = True

seed: int = 30
np.random.seed(seed)

# オンラインTV-SEM のパラメータ
P: int = 1
C: int = 1
gamma: float = 0.999
alpha: float = 0.015
beta_pc: float = 0.015
beta_co: float = 0.015
beta_sgd: float = 0.015

# その他のパラメータ（Proposed 手法用）
r: int = 4   # window size
q: int = 10  # number of processors
rho: float = 0.15
mu_lambda: float = 1

# 各手法の実行フラグ
run_pc_flag: bool = True     # Prediction Correction
run_co_flag: bool = True     # Correction Only
run_sgd_flag: bool = True    # SGD
run_pp_flag: bool = True     # Proposed

# 結果保存用ディレクトリの作成
today_str: str = datetime.datetime.now().strftime('%y%m%d')
result_dir: str = f'./result/{today_str}/images'
os.makedirs(result_dir, exist_ok=True)

# ----- 繰り返しシミュレーションループ -----
for sim in range(num_repeats):
    print(f"--- Simulation run {sim+1}/{num_repeats} ---")
    
    # 1. 【generate_linear_X】を用いて線形補間によるデータ生成
    S_series, X = generate_linear_X(
        N=N,
        T=T,
        S_is_symmetric=S_is_symmetric,
        sparsity=sparsity,
        max_weight=max_weight,
        std_e=std_e
    )
    
    # 参考: t=0 の隣接行列の SNR を計算
    snr_before: float = calc_snr(S_series[0])
    print("SNR at t=0 (S_series[0]):", snr_before)
    
    # 2. 推定開始時の初期値 S0 の生成
    S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_0 = S_0 / norm(S_0)  # 正則化
    
    # 3. 各推定手法のモデルインスタンス生成
    tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, name="pc")
    tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, name="co")
    tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, name="sgd")
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, name="pp")
    
    # 4. 各手法の実行関数定義（後で並列実行）
    def run_tv_sem_pc() -> Tuple[List[np.ndarray], List[float]]:
        return tv_sem_pc.run(X)

    def run_tv_sem_co() -> Tuple[List[np.ndarray], List[float]]:
        return tv_sem_co.run(X)

    def run_tv_sem_sgd() -> Tuple[List[np.ndarray], List[float]]:
        return tv_sem_sgd.run(X)

    def run_tv_sem_pp() -> List[np.ndarray]:
        return tv_sem_pp.run(X)
    
    # 5. 並列処理のジョブリスト作成
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
    
    # 6. 各手法の結果の格納
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
    
    # 7. 時系列毎の推定誤差（NSE）の計算
    error_pc:  List[float] = []
    error_co:  List[float] = []
    error_sgd: List[float] = []
    error_pp:  List[float] = []
    
    for t in range(T):
        S_true = S_series[t]
        if run_pc_flag:
            err_val = (norm(estimates_pc[t] - S_true) ** 2) / (norm(S_0 - S_true) ** 2)
            error_pc.append(err_val)
        if run_co_flag:
            err_val = (norm(estimates_co[t] - S_true) ** 2) / (norm(S_0 - S_true) ** 2)
            error_co.append(err_val)
        if run_sgd_flag:
            err_val = (norm(estimates_sgd[t] - S_true) ** 2) / (norm(S_0 - S_true) ** 2)
            error_sgd.append(err_val)
        if run_pp_flag:
            err_val = (norm(estimates_pp[t] - S_true) ** 2) / (norm(S_0 - S_true) ** 2)
            error_pp.append(err_val)
    
    # 8. 推定誤差のプロット
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
    plt.xlim(0, T)
    plt.xlabel('t')
    plt.ylabel('NSE')
    plt.grid(True, which="both")
    plt.legend()
    
    timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename: str = (
        f'result_run{sim+1}_N{N}_T{T}_maxweight{max_weight}_'
        f'variancee{variance_e}_Ssym{S_is_symmetric}_seed{seed}_'
        f'P{P}_C{C}_gamma{gamma}_alpha{alpha}_'
        f'betapc{beta_pc}_betaco{beta_co}_betasgd{beta_sgd}_'
        f'r{r}_q{q}_rho{rho}_mu{mu_lambda}_timestamp{timestamp}.png'
    )
    
    plt.savefig(os.path.join(result_dir, filename))
    plt.show()
    
    # 9. （オプション）実行したコードのバックアップ
    try:
        notebook_filename: str = os.path.basename(__file__)
        copy_ipynb_path: str = os.path.join(result_dir, f"{notebook_filename}_backup_{timestamp}.py")
        shutil.copy(notebook_filename, copy_ipynb_path)
        print(f"Notebook file copied to: {copy_ipynb_path}")
    except NameError:
        # インタラクティブモードの場合 __file__ は定義されない
        pass

print("全シミュレーション終了")
