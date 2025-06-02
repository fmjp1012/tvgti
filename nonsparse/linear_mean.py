import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import datetime
import shutil
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from utils import *
from models.tvgti_pc_nonsparse import TimeVaryingSEM as TimeVaryingSEM_PC_NONSPARSE
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED

# ----------------------------
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
# ----------------------------

# 試行回数（例：10回に変更）
num_trials = 10  # 100 -> 10に変更

# --- シミュレーションパラメータ ---
N = 5  # 10 -> 5に変更
T = 100  # 10000 -> 100に変更
sparsity = 0.0
max_weight = 0.5
variance_e = 0.005
std_e = np.sqrt(variance_e)
S_is_symmetric = True

seed = 30  # 基本のシード（試行ごとにシードをずらします）

# TV-SEM関連パラメータ
P = 1
C = 1
gamma = 0.999
alpha = 0.015
beta_pc = 0.015
beta_co = 0.015
beta_sgd = 0.015

# その他のパラメータ（Proposed 手法用）
r = 4    # window size
q = 10   # 並列処理数（プロセッサ数）
rho = 0.15
mu_lambda = 1

# ----------------------------
# 1試行分のシミュレーションを実行する関数
def run_trial(trial_seed: int):
    np.random.seed(trial_seed)  # 試行ごとのシード設定

    # データ生成（generate_linear_X を使用）
    S_series, X = generate_linear_X(
        N=N,
        T=T,
        S_is_symmetric=S_is_symmetric,
        sparsity=sparsity,
        max_weight=max_weight,
        std_e=std_e
    )

    # 初期推定行列の設定
    S_0 = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_0 = S_0 / norm(S_0)

    # モデルのインスタンス化
    tv_sem_pc = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_pc, gamma, P, C, name="pc")
    tv_sem_co = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_co, gamma, 0, C, name="co")
    tv_sem_sgd = TimeVaryingSEM_PC_NONSPARSE(N, S_0, alpha, beta_sgd, 0, 0, C, name="sgd")
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, name="pp")

    # 各手法の実行（各 run() は、時系列全体の推定結果をリストで返す）
    estimates_pc, _ = tv_sem_pc.run(X)
    estimates_co, _ = tv_sem_co.run(X)
    estimates_sgd, _ = tv_sem_sgd.run(X)
    estimates_pp = tv_sem_pp.run(X)

    # 各時刻 t における相対誤差（NSE）の計算
    error_pc = np.zeros(T)
    error_co = np.zeros(T)
    error_sgd = np.zeros(T)
    error_pp = np.zeros(T)

    for t in range(T):
        S_true = S_series[t]
        error_pc[t] = (norm(estimates_pc[t] - S_true)**2) / (norm(S_0 - S_true)**2)
        error_co[t] = (norm(estimates_co[t] - S_true)**2) / (norm(S_0 - S_true)**2)
        error_sgd[t] = (norm(estimates_sgd[t] - S_true)**2) / (norm(S_0 - S_true)**2)
        error_pp[t] = (norm(estimates_pp[t] - S_true)**2) / (norm(S_0 - S_true)**2)

    return error_pc, error_co, error_sgd, error_pp

# ----------------------------
# 試行ごとのシードリストを作成
trial_seeds = [seed + i for i in range(num_trials)]

# tqdm_joblib を用いて並列処理の進捗表示
with tqdm_joblib(tqdm(desc="Progress", total=num_trials)) as progress_bar:
    results = Parallel(n_jobs=-1, batch_size=1)(
        delayed(run_trial)(ts) for ts in trial_seeds
    )

# 各手法のエラーを試行ごとに集計
error_pc_total = np.zeros(T)
error_co_total = np.zeros(T)
error_sgd_total = np.zeros(T)
error_pp_total = np.zeros(T)

for error_pc, error_co, error_sgd, error_pp in results:
    error_pc_total += error_pc
    error_co_total += error_co
    error_sgd_total += error_sgd
    error_pp_total += error_pp

# 平均値の計算
error_pc_mean = error_pc_total / num_trials
error_co_mean = error_co_total / num_trials
error_sgd_mean = error_sgd_total / num_trials
error_pp_mean = error_pp_total / num_trials

# ----------------------------
# 結果のプロット
plt.figure(figsize=(10,6))
plt.plot(error_co_mean, color='blue', label='Correction Only')
plt.plot(error_pc_mean, color='limegreen', label='Prediction Correction')
plt.plot(error_sgd_mean, color='cyan', label='SGD')
plt.plot(error_pp_mean, color='red', label='Proposed')
plt.yscale('log')
plt.xlim(0, T)
plt.xlabel('t')
plt.ylabel('Average NSE')
plt.grid(True, which='both')
plt.legend()

# 結果画像の保存設定
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# __file__ が定義されていない場合は適宜ファイル名を指定してください
notebook_filename = os.path.basename(__file__) if '__file__' in globals() else 'simulation'
filename = (
    f'timestamp{timestamp}_'
    f'result_N{N}_'
    f'notebook_filename{notebook_filename}_'
    f'num_trials{num_trials}_'
    f'T{T}_'
    f'maxweight{max_weight}_'
    f'variancee{variance_e}_'
    f'Sissymmetric{S_is_symmetric}_'
    f'seed{seed}_'
    f'P{P}_'
    f'C{C}_'
    f'gamma{gamma}_'
    f'alpha{alpha}_'
    f'betapc{beta_pc}_'
    f'betaco{beta_co}_'
    f'betasgd{beta_sgd}_'
    f'r{r}_'
    f'q{q}_'
    f'rho{rho}_'
    f'mu_lambda{mu_lambda}.png'
)

today_str = datetime.datetime.now().strftime('%y%m%d')
save_path = os.path.join('.', 'result', today_str, 'images')
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

# ノートブック（またはスクリプト）のバックアップコピーを保存
copy_ipynb_path = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(__file__, copy_ipynb_path)  # notebook_filename -> __file__に変更
print(f"Notebook file copied to: {copy_ipynb_path}")
