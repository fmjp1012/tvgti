import shutil
import sys
import os
import datetime
from typing import List, Tuple, Dict

import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager

# ここでは utils などから下記の関数を import する想定:
# - generate_piecewise_X_K_with_snr(N, T, S_is_symmetric, sparsity, max_weight, std_e, K, snr_target)
# - solve_offline_sem(X, lambda_)  (lambda_ は 0 で呼んでいる)
# - calc_snr(S)  (必要ならば)
# - あるいは NSE を直接計算する関数 (なければ自作)

from utils import generate_piecewise_X_K_with_snr, solve_offline_sem

# ----------------------------------------------------
# matplotlib の設定 (必要に応じて)
# ----------------------------------------------------
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

# ----------------------------------------------------
# 固定パラメータの設定
# ----------------------------------------------------
N: int = 10
T: int = 5000
sparsity: float = 0.0
max_weight: float = 0.5
variance_e: float = 0.005
std_e: float = np.sqrt(variance_e)
K: int = 1
S_is_symmetric: bool = True

# NSE 計算用の関数 (utils にある想定がなければ自作)
# 例: Frobenius ノルムを使った誤差の割合
def calc_nse(S_true: np.ndarray, S_est: np.ndarray) -> float:
    return (norm(S_true - S_est, 'fro') ** 2) / (norm(S_true, 'fro') ** 2)

# 並列実行する処理を関数化
def run_experiment_for_snr(snr_target: float, seed: int = 10) -> float:
    """
    指定された SNR でデータを生成し、オフライン解を求め、その NSE を返す。
    """
    # SNR 毎に seed を変えたい場合は以下のようにする (任意)
    # 例: seed + int(snr_target)
    np.random.seed(seed + int(snr_target))

    # データ生成 (S_series[0] を真の行列と想定)
    S_series, X = generate_piecewise_X_K_with_snr(
        N,
        T,
        S_is_symmetric,
        sparsity,
        max_weight,
        std_e,
        K,
        snr_target
    )
    S_true = S_series[0]  # 今回は1区間として S_series[0] を真の行列とみなす

    # オフライン推定
    S_opt = solve_offline_sem(X, 0)  # 正則化パラメータは 0 とする

    # NSE を計算
    nse = calc_nse(S_true, S_opt)

    return nse

if __name__ == "__main__":
    # ----------------------------------------------------
    # 計算
    # ----------------------------------------------------
    snr_values = range(1, 21)  # 1 〜 20
    seed = 10  # 再現性のための固定シード(必要に応じて)

    # 並列実行 (n_jobs=-1 で CPU 全コア使用)
    nse_list = Parallel(n_jobs=-1)(
        delayed(run_experiment_for_snr)(snr, seed) 
        for snr in snr_values
    )

    # ----------------------------------------------------
    # 結果の可視化
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(snr_values, nse_list, marker="o", label="NSE vs. SNR")
    plt.xlabel("SNR")
    plt.ylabel("NSE (log scale)")
    plt.yscale('log')       # NSE を対数表示したい場合
    plt.grid(True, which="both")
    plt.legend()

    # ファイル名
    timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    notebook_filename: str = os.path.basename(__file__)
    filename: str = (
        f"result_snr_vs_nse_"
        f"N{N}_T{T}_"
        f"maxweight{max_weight}_variancee{variance_e}_"
        f"K{K}_Sissymmetric{S_is_symmetric}_"
        f"seed{seed}_"
        f"timestamp{timestamp}.png"
    )
    today_str: str = datetime.datetime.now().strftime('%y%m%d')
    save_path: str = f'./result/{today_str}/images'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename))
    plt.show()

    # ノートブック(or .py)ファイルのバックアップ（任意）
    copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
    shutil.copy(notebook_filename, copy_ipynb_path)
    print(f"Notebook file copied to: {copy_ipynb_path}")
