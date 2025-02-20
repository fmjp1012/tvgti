import shutil
import os
import datetime

import numpy as np
from scipy.linalg import inv, eigvals, norm
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from multiprocessing import Manager

from utils import *
from models.tvgti_pc_nonsparse import TimeVaryingSEM as TimeVaryingSEM_PC_NONSPARSE
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED

# プロットの設定
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

#-------------------------
# 手法ごとの実行スイッチフラグ
run_pc_flag = False     # Prediction Correction（PC）
run_co_flag = False     # Correction Only（CO）
run_sgd_flag = True    # SGD
run_pp_flag = True     # Proposed（PP）
#-------------------------

# 試行回数・パラメータの設定
num_trials = 10
N = 30
T = 3000
sparsity = 0
max_weight = 0.5
variance_e = 0.005
std_e = np.sqrt(variance_e)
K = 1
S_is_symmetric = True

seed = 3  # 基本シード

# TV-SEMパラメータ
P = 1
C = 1
gamma = 0.999
alpha = 0.015
beta_pc = 0.015
beta_co = 0.02
beta_sgd = 0.0269

# その他のパラメータ
r = 1  # window size
q = 1  # number of processors
rho = 0.0641
mu_lambda = 0.1

def run_trial(trial_seed):
    np.random.seed(trial_seed)  # 試行ごとにシード設定

    # データの生成
    S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)
    
    # 初期値の設定
    S_0 = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_0 = S_0 / norm(S_0)

    errors = {}

    # Prediction Correction
    if run_pc_flag:
        tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, False)
        estimates_pc, cost_values_pc = tv_sem_pc.run(X)
        error_pc = [ (norm(estimates_pc[i] - S_series[i])**2) / (norm(S_0 - S_series[i])**2)
                     for i in range(T) ]
        errors['pc'] = error_pc

    # Correction Only
    if run_co_flag:
        tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, False)
        estimates_co, cost_values_co = tv_sem_co.run(X)
        error_co = [ (norm(estimates_co[i] - S_series[i])**2) / (norm(S_0 - S_series[i])**2)
                     for i in range(T) ]
        errors['co'] = error_co

    # SGD
    if run_sgd_flag:
        tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, False)
        estimates_sgd, cost_values_sgd = tv_sem_sgd.run(X)
        error_sgd = [ (norm(estimates_sgd[i] - S_series[i])**2) / (norm(S_0 - S_series[i])**2)
                      for i in range(T) ]
        errors['sgd'] = error_sgd

    # Proposed
    if run_pp_flag:
        tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, False)
        estimates_pp = tv_sem_pp.run(X)
        error_pp = [ (norm(estimates_pp[i] - S_series[i])**2) / (norm(S_0 - S_series[i])**2)
                     for i in range(T) ]
        errors['pp'] = error_pp

    return errors

# 試行ごとのシードを作成
trial_seeds = [seed + i for i in range(num_trials)]

# 並列処理で各試行を実行（tqdm_joblib により進捗表示）
with tqdm_joblib(tqdm(desc="Progress", total=num_trials)) as progress_bar:
    results = Parallel(n_jobs=-1, batch_size=1)(
        delayed(run_trial)(trial_seed) for trial_seed in trial_seeds
    )

# 各手法ごとの誤差の合計（T×1の配列）を初期化
error_pc_total = np.zeros(T) if run_pc_flag else None
error_co_total = np.zeros(T) if run_co_flag else None
error_sgd_total = np.zeros(T) if run_sgd_flag else None
error_pp_total = np.zeros(T) if run_pp_flag else None

# 試行ごとに結果を集計
for errors in results:
    if run_pc_flag:
        error_pc_total += np.array(errors['pc'])
    if run_co_flag:
        error_co_total += np.array(errors['co'])
    if run_sgd_flag:
        error_sgd_total += np.array(errors['sgd'])
    if run_pp_flag:
        error_pp_total += np.array(errors['pp'])

# 試行ごとの平均誤差を計算
if run_pc_flag:
    error_pc_mean = error_pc_total / num_trials
if run_co_flag:
    error_co_mean = error_co_total / num_trials
if run_sgd_flag:
    error_sgd_mean = error_sgd_total / num_trials
if run_pp_flag:
    error_pp_mean = error_pp_total / num_trials

# 結果のプロット
plt.figure(figsize=(10, 6))
if run_co_flag:
    plt.plot(error_co_mean, color='blue', label='Correction Only')
if run_pc_flag:
    plt.plot(error_pc_mean, color='limegreen', label='Prediction Correction')
if run_sgd_flag:
    plt.plot(error_sgd_mean, color='cyan', label='SGD')
if run_pp_flag:
    plt.plot(error_pp_mean, color='red', label='Proposed')

plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('Average NSE')
plt.grid(True, 'both')
plt.legend()

# ファイル名の生成と保存先の設定
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
notebook_filename = os.path.basename(__file__)
filename = (
    f'result_N{N}_'
    f'notebook_filename{notebook_filename}_'
    f'num_trials{num_trials}_'
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
    f'timestamp{timestamp}_.png'
)
print(filename)
today_str = datetime.datetime.now().strftime('%y%m%d')
save_path = os.path.join('./result', today_str, 'images')
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

copy_ipynb_path = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
