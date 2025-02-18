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
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED
from models.tvgti_pp_nonsparse_undirected_nonoverlap import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED_NONOVERLAP

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

# 初期値の設定
S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
S_0 = S_0 / norm(S_0)

# その他のパラメータ
r: int = 300  # window size
q: int = 100  # number of processors
rho: float = 10  # 試行回数の設定
mu_lambda: float = 1

# モデルのインスタンス化
tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(N, S_0, r, q, rho, mu_lambda, name="pp")
tv_sem_pp_nonoverlap = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED_NONOVERLAP(N, S_0, r, q, rho, mu_lambda, name="pp_nonoverlap")

# 実行関数の定義
def run_tv_sem_pp() -> List[np.ndarray]:
    estimates_pp = tv_sem_pp.run(X)
    return estimates_pp

def run_tv_sem_pp_nonoverlap() -> List[np.ndarray]:
    estimates_pp_nonoverlap = tv_sem_pp_nonoverlap.run(X)
    return estimates_pp_nonoverlap


#----------------------------------------------------
# ここで実行対象の関数だけリストを作る
# (関数と手法名をまとめて管理するとあとで集計しやすい)
job_list = []
job_list.append(delayed(run_tv_sem_pp)())
job_list.append(delayed(run_tv_sem_pp_nonoverlap)())

results = Parallel(n_jobs=2)(job_list)
#----------------------------------------------------

# それぞれの結果を受け取る格納リスト
# （実行しないメソッドは空のままにする）
estimates_pp: List[np.ndarray] = []
estimates_pp_nonoverlap: List[np.ndarray] = []

# 実行した順番に応じて results から取り出す
idx_result: int = 0

estimates_pp = results[idx_result]
idx_result += 1
estimates_pp_nonoverlap = results[idx_result]
idx_result += 1

# ここから結果の解析・可視化
# （実行したメソッドだけ処理をする）
S_opts: List[np.ndarray] = []
NSE_pp: List[float] = []
NSE_pp_nonoverlap: List[float] = []
error_pp: List[float] = []
error_pp_nonoverlap: List[float] = []

sum_error_pp: List[float] = []
sum_error_pp_non_overlap: List[float] = []

for i, estimate in enumerate(estimates_pp):
    error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
    error_pp.append(error_val)
    sum_error_pp.append((estimate - S_series[i]).sum())

for i, estimate in enumerate(estimates_pp_nonoverlap):
    error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
    error_pp_nonoverlap.append(error_val)
    sum_error_pp_non_overlap.append((estimate - S_series[i]).sum())

# 結果のプロット
plt.figure(figsize=(10,6))

plt.plot(error_pp, color='red', label='Proposed')
plt.plot(error_pp_nonoverlap, color='orange', label='Proposed_nonoverlap')

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
