# %%
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

# 試行回数の設定
num_trials = 10

# パラメータの設定
N = 10
T = 20000
sparsity = 100
max_weight = 0.5
variance_e = 0.005
std_e = np.sqrt(variance_e)
K = 4
S_is_symmetric = True

seed = 42  # 基本のシード

# TV-SEMパラメータ
P = 1
C = 1
gamma = 0.999
alpha = 0.015
beta_pc = 0.015
beta_co = 0.02
beta_sgd = 0.02

# その他のパラメータ
r = 4  # window size
q = 20  # number of processors
rho = 0.15
mu_lambda = 0.5

def run_trial(trial_seed):
    np.random.seed(trial_seed)  # シードを試行ごとに設定

    # データの生成
    S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)

    # 初期値の設定
    S_0 = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_0 = S_0 / norm(S_0)

    # モデルのインスタンス化（並列処理を無効化したい場合は注意）
    tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, False)
    tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, False)
    tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, False)
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, False)

    # 並列処理を無効化するために直接呼び出しているが、Parallel( n_jobs=... ) を使ってもOK
    estimates_pc, cost_values_pc = tv_sem_pc.run(X)
    estimates_co, cost_values_co = tv_sem_co.run(X)
    estimates_sgd, cost_values_sgd = tv_sem_sgd.run(X)
    estimates_pp = tv_sem_pp.run(X)

    # エラーの計算
    error_pc = []
    error_co = []
    error_sgd = []
    error_pp = []

    for i in range(T):
        error_pc.append(norm(estimates_pc[i] - S_series[i]) ** 2 / (norm(S_0 - S_series[i]) ** 2))
        error_co.append(norm(estimates_co[i] - S_series[i]) ** 2 / (norm(S_0 - S_series[i]) ** 2))
        error_sgd.append(norm(estimates_sgd[i] - S_series[i]) ** 2 / (norm(S_0 - S_series[i]) ** 2))
        error_pp.append(norm(estimates_pp[i] - S_series[i]) ** 2 / (norm(S_0 - S_series[i]) ** 2))

    return error_pc, error_co, error_sgd, error_pp

# 試行ごとのシードを作成
trial_seeds = [seed + i for i in range(num_trials)]

# tqdm_joblib を使って Parallel の処理に進捗表示を付ける
with tqdm_joblib(tqdm(desc="Progress", total=num_trials)) as progress_bar:
    results = Parallel(n_jobs=-1, batch_size=1)(
        delayed(run_trial)(trial_seed) for trial_seed in trial_seeds
    )

# 結果の集計
error_pc_total = np.zeros(T)
error_co_total = np.zeros(T)
error_sgd_total = np.zeros(T)
error_pp_total = np.zeros(T)

for error_pc, error_co, error_sgd, error_pp in results:
    error_pc_total += np.array(error_pc)
    error_co_total += np.array(error_co)
    error_sgd_total += np.array(error_sgd)
    error_pp_total += np.array(error_pp)

# 平均の計算
error_pc_mean = error_pc_total / num_trials
error_co_mean = error_co_total / num_trials
error_sgd_mean = error_sgd_total / num_trials
error_pp_mean = error_pp_total / num_trials

# 結果のプロット
plt.figure(figsize=(10,6))
plt.plot(error_co_mean, color='blue', label='Correction Only')
plt.plot(error_pc_mean, color='limegreen', label='Prediction Correction')
plt.plot(error_sgd_mean, color='cyan', label='SGD')
plt.plot(error_pp_mean, color='red', label='Proposed')
plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('Average NSE')
plt.grid(True, 'both')
plt.legend()

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

notebook_filename = "sandbox_mean.py"  # ★使用中のNotebook名を入力

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
save_path = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)  # ディレクトリが無い場合は作成
plt.savefig(os.path.join(save_path, filename))
plt.show()

copy_ipynb_path = os.path.join(save_path, f"sandbox_mean_backup_{timestamp}.py")

shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
