import shutil
import sys
import os
import datetime

import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from utils import *
from models.tvgti_pp_nonsparse_undirected import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED
from models.tvgti_pp_nonsparse_undirected_nonoverlap import TimeVaryingSEM as TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED_NONOVERLAP

# プロット設定
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

#--------------------------
# シミュレーションパラメータ
#--------------------------
num_trials = 100  # 試行回数

N: int = 30
T: int = 3000
sparsity: float = 0     # 辺のスパース性（今回は0:全結合？）
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True

# 各手法用パラメータ
# Proposed 手法
r_pp: int = 30       # ウィンドウサイズ
q_pp: int = 20       # プロセッサ数
rho_pp: float = 2.48
mu_lambda_pp: float = 1

# Proposed_nonoverlap 手法
r_pp_nonoverlap: int = 30
q_pp_nonoverlap: int = 20
rho_pp_nonoverlap: float = 2.48
mu_lambda_pp_nonoverlap: float = 1

base_seed: int = 3  # 試行ごとのシードは base_seed + i

#--------------------------------------------
# 1回の試行を実行する関数
#--------------------------------------------
def run_trial(trial_seed: int):
    # 試行ごとにシードを設定
    np.random.seed(trial_seed)
    
    # データ生成（S_series: 真のグラフ系列, X: 観測データ）
    S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K)
    
    # 初期値の設定
    S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_0 = S_0 / norm(S_0)
    
    # 各手法のインスタンス作成
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(
        N, S_0, r_pp, q_pp, rho_pp, mu_lambda_pp, name="pp"
    )
    tv_sem_pp_nonoverlap = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED_NONOVERLAP(
        N, S_0, r_pp_nonoverlap, q_pp_nonoverlap, rho_pp_nonoverlap, mu_lambda_pp_nonoverlap, name="pp_nonoverlap"
    )
    
    # 各手法の実行（推定系列を取得）
    estimates_pp = tv_sem_pp.run(X)
    estimates_pp_nonoverlap = tv_sem_pp_nonoverlap.run(X)
    
    # 時刻ごとにNSEを計算
    error_pp = []
    error_pp_nonoverlap = []
    for i in range(T):
        err_pp = (norm(estimates_pp[i] - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        err_pp_nonoverlap = (norm(estimates_pp_nonoverlap[i] - S_series[i]) ** 2) / (norm(S_0 - S_series[i]) ** 2)
        error_pp.append(err_pp)
        error_pp_nonoverlap.append(err_pp_nonoverlap)
    
    return error_pp, error_pp_nonoverlap

#--------------------------------------------
# 複数試行の実行（並列処理＋進捗表示）
#--------------------------------------------
trial_seeds = [base_seed + i for i in range(num_trials)]
with tqdm_joblib(tqdm(desc="Trials", total=num_trials)) as progress_bar:
    trial_results = Parallel(n_jobs=-1)(delayed(run_trial)(seed) for seed in trial_seeds)

# 各手法のエラーを時刻ごとに累積
error_pp_total = np.zeros(T)
error_pp_nonoverlap_total = np.zeros(T)

for err_pp, err_pp_nonoverlap in trial_results:
    error_pp_total += np.array(err_pp)
    error_pp_nonoverlap_total += np.array(err_pp_nonoverlap)

# 平均エラーの計算
error_pp_mean = error_pp_total / num_trials
error_pp_nonoverlap_mean = error_pp_nonoverlap_total / num_trials

#--------------------------------------------
# 結果のプロット
#--------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(error_pp_mean, color='red', label='Proposed')
plt.plot(error_pp_nonoverlap_mean, color='orange', label='Proposed_nonoverlap')
plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('Average NSE')
plt.grid(True, which="both")
plt.legend()

# 結果保存用のファイル名生成
timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
notebook_filename: str = os.path.basename(__file__)
filename: str = (
    f'timestamp{timestamp}_'
    f'result_N{N}_'
    f'{notebook_filename}_'
    f'T{T}_'
    f'maxweight{max_weight}_'
    f'variancee{variance_e}_'
    f'K{K}_'
    f'Sissymmetric{S_is_symmetric}_'
    f'num_trials{num_trials}_'
    f'seed{base_seed}_'
    f'r_pp{r_pp}_{r_pp_nonoverlap}_'
    f'q_pp{q_pp}_{q_pp_nonoverlap}_'
    f'rho_pp{rho_pp}_{rho_pp_nonoverlap}_'
    f'mu_lambda_pp{mu_lambda_pp}_{mu_lambda_pp_nonoverlap}.png'
)

today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = os.path.join('.', 'result', today_str, 'images')
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

# 元のスクリプトのバックアップ保存
copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(notebook_filename, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}")
