import shutil
import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import datetime
from typing import List, Tuple, Dict

import numpy as np
from scipy.linalg import inv, eigvals, norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager
import optuna

from utils import *
from models.tvgti_pc import TimeVaryingSEM as TimeVaryingSEM_PC_SPARSE
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

#----------------------------------------------------
# メソッドごとの実行スイッチ
run_pc_flag: bool = False     # Prediction Correction
run_co_flag: bool = False     # Correction Only
run_sgd_flag: bool = True     # SGD
run_pp_flag: bool = True      # Proposed (ハイパーパラメータチューニング付き)
#----------------------------------------------------

# ハイパーパラメータチューニングの設定
enable_hyperparameter_tuning: bool = True  # PPメソッドのハイパーパラメータチューニングを有効にする
enable_sgd_hyperparameter_tuning: bool = True  # SGDメソッドのハイパーパラメータチューニングを有効にする
n_tuning_trials: int = 10  # チューニングの試行回数

# パラメータの設定
N: int = 50
T: int = 1000
sparsity: float = 0.5  # sparse version uses non-zero sparsity
max_weight: float = 0.5
variance_e: float = 0.005  # adjusted for sparse version
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True

seed: int = 3
np.random.seed(seed)

# TV-SEMシミュレーション
S_series, X = generate_piecewise_X_K(N, T, S_is_symmetric, sparsity, max_weight, std_e, K, s_type="regular")

snr_before: float = calc_snr(S_series[0])
print("Before scaling, SNR =", snr_before)

# オンラインTV-SEMパラメータ
P: int = 1
C: int = 1
gamma: float = 0.999
alpha: float = 0.02
beta_pc: float = 0.02
beta_co: float = 0.02
beta_sgd: float = 0.0269
lambda_reg: float = 0.01  # regularization parameter for sparse models

# 初期値の設定
S_0: np.ndarray = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
S_0 = S_0 / norm(S_0)

# PPメソッドのデフォルトパラメータ
r: int = 1  # window size
q: int = 1  # number of processors
rho: float = 0.0641
mu_lambda: float = 0.1

#----------------------------------------------------
# ハイパーパラメータチューニング関数
#----------------------------------------------------

def objective_function(trial: optuna.trial.Trial) -> float:
    """
    PPメソッドのハイパーパラメータ最適化のための目的関数
    最終時刻のNSEを最小化する
    """
    # ハイパーパラメータのサンプリング
    rho_suggested = trial.suggest_float("rho", 1e-6, 0.1, log=False)
    mu_lambda_suggested = trial.suggest_float("mu_lambda", 1e-6, 0.1, log=False)
    
    # PPメソッドを実行
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(
        N, S_0, r, q, rho_suggested, mu_lambda_suggested, name="pp_tuning"
    )
    estimates_pp = tv_sem_pp.run(X)
    
    # 最終時刻のNSEを計算
    final_estimate = estimates_pp[-1]
    final_true = S_series[-1]
    final_nse = (norm(final_estimate - final_true) ** 2) / (norm(S_0 - final_true) ** 2)
    
    return final_nse

def objective_function_sgd(trial: optuna.trial.Trial) -> float:
    """
    SGDメソッドのハイパーパラメータ最適化のための目的関数
    最終時刻のNSEを最小化する
    """
    # ハイパーパラメータのサンプリング
    beta_sgd_suggested = trial.suggest_float("beta_sgd", 1e-6, 0.1, log=False)
    lambda_reg_suggested = trial.suggest_float("lambda_reg", 1e-6, 0.1, log=False)
    
    # SGDメソッドを実行
    tv_sem_sgd_tuning = TimeVaryingSEM_PC_SPARSE(
        N, S_0, lambda_reg_suggested, alpha, beta_sgd_suggested, 0, 0, C, name="sgd_tuning"
    )
    estimates_sgd_tuning = tv_sem_sgd_tuning.run(X)[0]  # [0]でestimatesのみ取得
    
    # 最終時刻のNSEを計算
    final_estimate = estimates_sgd_tuning[-1]
    final_true = S_series[-1]
    final_nse = (norm(final_estimate - final_true) ** 2) / (norm(S_0 - final_true) ** 2)
    
    return final_nse

def tune_pp_hyperparameters() -> Tuple[float, float]:
    """
    PPメソッドのハイパーパラメータをチューニングする
    Returns:
        Tuple[float, float]: 最適化されたrhoとmu_lambdaの値
    """
    print("PPメソッドのハイパーパラメータチューニングを開始...")
    
    # Optunaで最適化
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_function, n_trials=n_tuning_trials)
    
    best_trial = study.best_trial
    print(f"チューニング完了:")
    print(f"  最適パラメータ: {best_trial.params}")
    print(f"  最終NSE: {best_trial.value}")
    
    best_rho = best_trial.params["rho"]
    best_mu_lambda = best_trial.params["mu_lambda"]
    
    return best_rho, best_mu_lambda

def tune_sgd_hyperparameters() -> Tuple[float, float]:
    """
    SGDメソッドのハイパーパラメータをチューニングする
    Returns:
        Tuple[float, float]: 最適化されたbeta_sgdとlambda_regの値
    """
    print("SGDメソッドのハイパーパラメータチューニングを開始...")
    
    # Optunaで最適化
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_function_sgd, n_trials=n_tuning_trials)
    
    best_trial = study.best_trial
    print(f"チューニング完了:")
    print(f"  最適パラメータ: {best_trial.params}")
    print(f"  最終NSE: {best_trial.value}")
    
    best_beta_sgd = best_trial.params["beta_sgd"]
    best_lambda_reg = best_trial.params["lambda_reg"]
    
    return best_beta_sgd, best_lambda_reg

#----------------------------------------------------
# 実行関数の定義
#----------------------------------------------------

# モデルのインスタンス化（PP、SGD以外）
tv_sem_pc = TimeVaryingSEM_PC_SPARSE(N, S_0, lambda_reg, alpha, beta_pc, gamma, P, C, name="pc")
tv_sem_co = TimeVaryingSEM_PC_SPARSE(N, S_0, lambda_reg, alpha, beta_co, gamma, 0, C, name="co")

def run_tv_sem_pc() -> Tuple[List[np.ndarray], List[float]]:
    estimates_pc, cost_values_pc = tv_sem_pc.run(X)
    return estimates_pc, cost_values_pc

def run_tv_sem_co() -> Tuple[List[np.ndarray], List[float]]:
    estimates_co, cost_values_co = tv_sem_co.run(X)
    return estimates_co, cost_values_co

def run_tv_sem_sgd(beta_sgd_param: float, lambda_reg_param: float) -> Tuple[List[np.ndarray], List[float]]:
    tv_sem_sgd = TimeVaryingSEM_PC_SPARSE(N, S_0, lambda_reg_param, alpha, beta_sgd_param, 0, 0, C, name="sgd")
    estimates_sgd, cost_values_sgd = tv_sem_sgd.run(X)
    return estimates_sgd, cost_values_sgd

def run_tv_sem_pp(rho_param: float, mu_lambda_param: float) -> List[np.ndarray]:
    tv_sem_pp = TimeVaryingSEM_PP_NONSPARSE_UNDIRECTED(
        N, S_0, r, q, rho_param, mu_lambda_param, name="pp"
    )
    estimates_pp = tv_sem_pp.run(X)
    return estimates_pp

#----------------------------------------------------
# ハイパーパラメータチューニング実行（PPメソッドが有効な場合のみ）
#----------------------------------------------------

if run_pp_flag and enable_hyperparameter_tuning:
    optimized_rho, optimized_mu_lambda = tune_pp_hyperparameters()
    print(f"最適化されたパラメータを使用: rho={optimized_rho:.6f}, mu_lambda={optimized_mu_lambda:.6f}")
else:
    # デフォルトパラメータを使用
    optimized_rho = rho
    optimized_mu_lambda = mu_lambda
    if run_pp_flag:
        print(f"デフォルトパラメータを使用: rho={optimized_rho:.6f}, mu_lambda={optimized_mu_lambda:.6f}")

#----------------------------------------------------
# ハイパーパラメータチューニング実行（SGDメソッドが有効な場合のみ）
#----------------------------------------------------

if run_sgd_flag and enable_sgd_hyperparameter_tuning:
    optimized_beta_sgd, optimized_lambda_reg = tune_sgd_hyperparameters()
    print(f"最適化されたパラメータを使用: beta_sgd={optimized_beta_sgd:.6f}, lambda_reg={optimized_lambda_reg:.6f}")
else:
    # デフォルトパラメータを使用
    optimized_beta_sgd = beta_sgd
    optimized_lambda_reg = lambda_reg
    if run_sgd_flag:
        print(f"デフォルトパラメータを使用: beta_sgd={optimized_beta_sgd:.6f}, lambda_reg={optimized_lambda_reg:.6f}")

#----------------------------------------------------
# 並列実行
#----------------------------------------------------

job_list = []
if run_pc_flag:
    job_list.append(delayed(run_tv_sem_pc)())
if run_co_flag:
    job_list.append(delayed(run_tv_sem_co)())
if run_sgd_flag:
    job_list.append(delayed(run_tv_sem_sgd)(optimized_beta_sgd, optimized_lambda_reg))
if run_pp_flag:
    job_list.append(delayed(run_tv_sem_pp)(optimized_rho, optimized_mu_lambda))

results = Parallel(n_jobs=4)(job_list)

#----------------------------------------------------
# 結果の取得
#----------------------------------------------------

# それぞれの結果を受け取る格納リスト
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
    estimates_pp = results[idx_result]
    idx_result += 1

#----------------------------------------------------
# 結果の解析・可視化
#----------------------------------------------------

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
    sgd_label = 'SGD (Tuned)' if enable_sgd_hyperparameter_tuning else 'SGD'
    plt.plot(error_sgd, color='cyan', label=sgd_label)
# Proposed
if run_pp_flag:
    tuning_label = 'Proposed (Tuned)' if enable_hyperparameter_tuning else 'Proposed'
    plt.plot(error_pp, color='red', label=tuning_label)

plt.yscale('log')
plt.xlim(left=0, right=T)
plt.xlabel('t')
plt.ylabel('NSE')
plt.grid(True, "both")
plt.legend()

timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
notebook_filename: str = os.path.basename(__file__)

tuning_suffix = ""
if run_pp_flag and enable_hyperparameter_tuning:
    tuning_suffix += "_pp_tuned"
if run_sgd_flag and enable_sgd_hyperparameter_tuning:
    tuning_suffix += "_sgd_tuned"

# SGDパラメータは最適化された値を使用
if run_sgd_flag:
    sgd_param_str = f'bsgd{optimized_beta_sgd:.6f}_lr{optimized_lambda_reg:.6f}'
else:
    sgd_param_str = f'bsgd{beta_sgd}_lr{lambda_reg}'

filename: str = (
    f'{timestamp}_'
    f'N{N}_T{T}_'
    f'mw{max_weight}_ve{variance_e}_'
    f'K{K}_sym{S_is_symmetric}_'
    f'sp{sparsity}_s{seed}_'
    f'P{P}_C{C}_g{gamma}_'
    f'a{alpha}_bpc{beta_pc}_'
    f'bco{beta_co}_{sgd_param_str}_'
    f'r{r}_'
    f'q{q}_rho{optimized_rho:.6f}_ml{optimized_mu_lambda:.6f}'
    f'{tuning_suffix}.png'
)

print(filename)
today_str: str = datetime.datetime.now().strftime('%y%m%d')
save_path: str = f'./result/{today_str}/images'
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, filename))
plt.show()

# ヒートマップ比較の追加
# 最終時点での真の隣接行列と推定結果を比較
final_time_idx = T - 1
true_S = S_series[final_time_idx]

# 実行された手法の数を数える
num_methods = sum([run_pc_flag, run_co_flag, run_sgd_flag, run_pp_flag])

if num_methods > 0:
    # サブプロット数を決定（真の行列 + 実行された手法数）
    total_plots = num_methods + 1
    cols = min(total_plots, 3)  # 最大3列
    rows = (total_plots + cols - 1) // cols  # 必要な行数
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if total_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    
    # 真の隣接行列
    ax = axes.flat[plot_idx] if total_plots > 1 else axes[plot_idx]
    im = ax.imshow(true_S, cmap='viridis', aspect='equal')
    ax.set_title('True Adjacency Matrix')
    plt.colorbar(im, ax=ax)
    plot_idx += 1
    
    # 各手法の推定結果
    if run_pc_flag and len(estimates_pc) > 0:
        ax = axes.flat[plot_idx]
        estimated_S = estimates_pc[final_time_idx]
        im = ax.imshow(estimated_S, cmap='viridis', aspect='equal')
        ax.set_title('PC Estimated Matrix')
        plt.colorbar(im, ax=ax)
        plot_idx += 1
    
    if run_co_flag and len(estimates_co) > 0:
        ax = axes.flat[plot_idx]
        estimated_S = estimates_co[final_time_idx]
        im = ax.imshow(estimated_S, cmap='viridis', aspect='equal')
        ax.set_title('CO Estimated Matrix')
        plt.colorbar(im, ax=ax)
        plot_idx += 1
    
    if run_sgd_flag and len(estimates_sgd) > 0:
        ax = axes.flat[plot_idx]
        estimated_S = estimates_sgd[final_time_idx]
        im = ax.imshow(estimated_S, cmap='viridis', aspect='equal')
        title_suffix = ' (Tuned)' if enable_sgd_hyperparameter_tuning else ''
        ax.set_title(f'SGD Estimated Matrix{title_suffix}')
        plt.colorbar(im, ax=ax)
        plot_idx += 1
    
    if run_pp_flag and len(estimates_pp) > 0:
        ax = axes.flat[plot_idx]
        estimated_S = estimates_pp[final_time_idx]
        im = ax.imshow(estimated_S, cmap='viridis', aspect='equal')
        title_suffix = ' (Tuned)' if enable_hyperparameter_tuning else ''
        ax.set_title(f'PP Estimated Matrix{title_suffix}')
        plt.colorbar(im, ax=ax)
        plot_idx += 1
    
    # 未使用のサブプロットを非表示
    for i in range(plot_idx, len(axes.flat)):
        axes.flat[i].set_visible(False)
    
    plt.tight_layout()
    
    # ヒートマップのファイル名
    heatmap_filename = filename.replace('.png', '_heatmap.png')
    plt.savefig(os.path.join(save_path, heatmap_filename))
    plt.show()
    
    print(f"Heatmap saved as: {heatmap_filename}")

# チューニング結果の保存
tuning_results = {
    'timestamp': timestamp,
    'n_trials': n_tuning_trials,
}

if run_pp_flag and enable_hyperparameter_tuning:
    tuning_results.update({
        'pp_optimized_rho': optimized_rho,
        'pp_optimized_mu_lambda': optimized_mu_lambda,
        'pp_final_nse': error_pp[-1] if error_pp else None
    })

if run_sgd_flag and enable_sgd_hyperparameter_tuning:
    tuning_results.update({
        'sgd_optimized_beta_sgd': optimized_beta_sgd,
        'sgd_optimized_lambda_reg': optimized_lambda_reg,
        'sgd_final_nse': error_sgd[-1] if error_sgd else None
    })

if (run_pp_flag and enable_hyperparameter_tuning) or (run_sgd_flag and enable_sgd_hyperparameter_tuning):
    import json
    tuning_results_path = os.path.join(save_path, f"tuning_results_{timestamp}.json")
    with open(tuning_results_path, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    print(f"Tuning results saved to: {tuning_results_path}")

copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
shutil.copy(__file__, copy_ipynb_path)
print(f"Notebook file copied to: {copy_ipynb_path}") 