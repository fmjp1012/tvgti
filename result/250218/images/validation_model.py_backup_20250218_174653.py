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

plt.rc('text', usetex=True)
plt.rc('font', family="serif")
plt.rcParams["font.family"] = "Times New Roman"      # 全体のフォントを設定
plt.rcParams["xtick.direction"] = "in"               # x軸の目盛線が内向き
plt.rcParams["ytick.direction"] = "in"               # y軸の目盛線が内向き
plt.rcParams["xtick.minor.visible"] = True          # x軸補助目盛りを表示
plt.rcParams["ytick.minor.visible"] = True          # y軸補助目盛りを表示
plt.rcParams["xtick.major.width"] = 1.5             # x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1.5             # y軸主目盛り線の線幅
plt.rcParams["xtick.minor.width"] = 1.0             # x軸補助目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 1.0             # y軸補助目盛り線の線幅
plt.rcParams["xtick.major.size"] = 10               # x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 10               # y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 5                # x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5                # y軸補助目盛り線の長さ
plt.rcParams["font.size"] = 15                      # フォントの大きさ

#----------------------------------------------------
# メソッドごとの実行スイッチ（True: 実行, False: スキップ）
#----------------------------------------------------
run_proposed_method = False
run_proposed_nonoverlap_method = True

# パラメータの設定
N: int = 30
T: int = 3000
sparsity: float = 0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
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

#----------------------------------------------------
# 2つの手法用にパラメータを別々に定義
# Proposed (pp) 手法用パラメータ
r_pp: int = 300       # window size
q_pp: int = 15       # number of processors
rho_pp: float = 10    # 試行回数の設定
mu_lambda_pp: float = 1

# Proposed_nonoverlap 手法用パラメータ
r_pp_nonoverlap: int = 300       # window size
q_pp_nonoverlap: int = 100       # number of processors
rho_pp_nonoverlap: float = 10    # 試行回数の設定
mu_lambda_pp_nonoverlap: float = 1

# モデルのインスタンス化
tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(
    N, S_0, r_pp, q_pp, rho_pp, mu_lambda_pp, name="pp"
)
tv_sem_pp_nonoverlap = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED_NONOVERLAP(
    N, S_0, r_pp_nonoverlap, q_pp_nonoverlap, rho_pp_nonoverlap, mu_lambda_pp_nonoverlap, name="pp_nonoverlap"
)

#----------------------------------------------------
# 実行関数の定義
def run_tv_sem_pp() -> List[np.ndarray]:
    estimates_pp = tv_sem_pp.run(X)
    return estimates_pp

def run_tv_sem_pp_nonoverlap() -> List[np.ndarray]:
    estimates_pp_nonoverlap = tv_sem_pp_nonoverlap.run(X)
    return estimates_pp_nonoverlap

#----------------------------------------------------
# 実行対象の関数をリストに追加（スイッチにより実行を制御）
job_list = []
if run_proposed_method:
    job_list.append(delayed(run_tv_sem_pp)())
if run_proposed_nonoverlap_method:
    job_list.append(delayed(run_tv_sem_pp_nonoverlap)())

if len(job_list) > 0:
    results = Parallel(n_jobs=len(job_list))(job_list)
else:
    results = []

# 結果格納用リストの初期化
estimates_pp: List[np.ndarray] = []
estimates_pp_nonoverlap: List[np.ndarray] = []

# 実行順に応じて results から取り出す
result_idx: int = 0
if run_proposed_method:
    estimates_pp = results[result_idx]
    result_idx += 1
if run_proposed_nonoverlap_method:
    estimates_pp_nonoverlap = results[result_idx]
    result_idx += 1

#----------------------------------------------------
# 解析・可視化のための変数定義
S_opts: List[np.ndarray] = []
error_pp: List[float] = []
error_pp_nonoverlap: List[float] = []

# Proposed 手法のエラー計算
if run_proposed_method:
    for i, estimate in enumerate(estimates_pp):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_pp.append(error_val)

# Proposed_nonoverlap 手法のエラー計算
if run_proposed_nonoverlap_method:
    for i, estimate in enumerate(estimates_pp_nonoverlap):
        error_val: float = (norm(estimate - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_pp_nonoverlap.append(error_val)

#----------------------------------------------------
# 結果のプロット
plt.figure(figsize=(10,6))

if run_proposed_method:
    plt.plot(error_pp, color='red', label='Proposed')
if run_proposed_nonoverlap_method:
    plt.plot(error_pp_nonoverlap, color='orange', label='Proposed_nonoverlap')

plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('NSE')
plt.grid(True, which="both")
plt.legend()

timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
notebook_filename: str = os.path.basename(__file__)

filename: str = (
    f'result_N{N}_'
    f'{notebook_filename}_'
    f'T{T}_'
    f'maxweight{max_weight}_'
    f'variancee{variance_e}_'
    f'K{K}_'
    f'Sissymmetric{S_is_symmetric}_'
    f'seed{seed}_'
    f'r_pp{r_pp}_{r_pp_nonoverlap}_'
    f'q_pp{q_pp}_{q_pp_nonoverlap}_'
    f'rho_pp{rho_pp}_{rho_pp_nonoverlap}_'
    f'mu_lambda_pp{mu_lambda_pp}_{mu_lambda_pp_nonoverlap}_'
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
