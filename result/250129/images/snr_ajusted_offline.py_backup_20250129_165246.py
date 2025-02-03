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

# ----------------------------------------------------
# 必要な関数のインポート (仮定)
# ----------------------------------------------------
# - generate_piecewise_X_K_with_snr(N, T, S_is_symmetric, sparsity, max_weight, std_e, K, snr_target)
# - solve_offline_sem(X, lambda_)
# - calc_snr(S)  <-- 「S から SNR を計算する」と仮定 (あるいはデータから計算するなら実装内容を調整)
from utils import generate_piecewise_X_K_with_snr, solve_offline_sem, calc_snr

# ----------------------------------------------------
# matplotlib の設定 (任意)
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

# ----------------------------------------------------
# NSE 計算用の関数
# ----------------------------------------------------
# 例: Frobenius ノルムを使った誤差の割合
def calc_nse(S_true: np.ndarray, S_est: np.ndarray) -> float:
    return (norm(S_true - S_est, 'fro') ** 2) / (norm(S_true, 'fro') ** 2)

# ----------------------------------------------------
# 並列実行する処理を関数化
# ----------------------------------------------------
def run_experiment_for_snr(snr_target: float, seed: int = 10) -> Tuple[float, float]:
    """
    指定された SNR でデータを生成し、オフライン解を求め:
      - NSE
      - 実測 SNR (calc_snr(...) の結果)
    を返す。
    """
    # SNR 毎にシードを変えたい場合は以下のようにする (任意)
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
    S_true = S_series[0]  # 今回は 1 区間として S_series[0] を真の行列とみなす

    # オフライン推定
    S_opt = solve_offline_sem(X, 0)  # 正則化パラメータは 0 とする

    # NSE を計算
    nse = calc_nse(S_true, S_opt)

    # 実測 SNR を計算
    # ここでは仮に calc_snr(S_true) を「実測」として扱う
    # （本当にデータ起因のノイズ比を測るなら、X と e を用いる実装が望ましい）
    snr_measured = calc_snr(S_true)

    return nse, snr_measured

# ----------------------------------------------------
# メイン処理
# ----------------------------------------------------
if __name__ == "__main__":
    # SNR の候補 (1 〜 20)
    snr_values = [1, 10, 100, 1000]
    seed = 10  # 再現性のための固定シード(必要に応じて)

    # 並列実行 (n_jobs=-1 で CPU 全コア使用)
    results = Parallel(n_jobs=-1)(
        delayed(run_experiment_for_snr)(snr, seed) 
        for snr in snr_values
    )

    # results は [(nse1, snr_measured1), (nse2, snr_measured2), ...] のリスト
    nse_list = [r[0] for r in results]
    snr_measured_list = [r[1] for r in results]

    # ----------------------------------------------------
    # 結果の可視化: SNR (target) vs NSE
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(snr_values, nse_list, marker="o", label="NSE vs. (target) SNR")
    plt.xlabel("Target SNR")
    plt.ylabel("NSE (log scale)")
    plt.yscale('log')
    plt.grid(True, which="both")
    plt.legend()
    plt.title("NSE vs. Target SNR")
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------
    # 結果の可視化: Target SNR vs Measured SNR (もし比較したい場合)
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(snr_values, snr_measured_list, marker="o", label="Measured SNR")
    plt.xlabel("Target SNR")
    plt.ylabel("Measured SNR (?)")
    plt.grid(True)
    plt.legend()
    plt.title("Target SNR vs. Measured SNR")
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------
    # ログ出力 (print) で確認
    # ----------------------------------------------------
    for snr_t, nse_val, snr_meas in zip(snr_values, nse_list, snr_measured_list):
        print(f"SNR target = {snr_t}, NSE = {nse_val:.4e}, SNR measured = {snr_meas:.4f}")

    # ----------------------------------------------------
    # 画像ファイルとして保存したい場合 (例)
    # ----------------------------------------------------
    timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    notebook_filename: str = os.path.basename(__file__)
    today_str: str = datetime.datetime.now().strftime('%y%m%d')
    save_path: str = f'./result/{today_str}/images'
    os.makedirs(save_path, exist_ok=True)

    # 1枚目 (NSE vs target SNR)
    filename_nse: str = (
        f"result_nse_vs_snr_"
        f"N{N}_T{T}_"
        f"maxweight{max_weight}_variancee{variance_e}_"
        f"K{K}_Sissymmetric{S_is_symmetric}_"
        f"seed{seed}_"
        f"timestamp{timestamp}.png"
    )
    plt.figure(figsize=(8, 6))
    plt.plot(snr_values, nse_list, marker="o", label="NSE vs. (target) SNR")
    plt.xlabel("Target SNR")
    plt.ylabel("NSE (log scale)")
    plt.yscale('log')
    plt.grid(True, which="both")
    plt.legend()
    plt.title("NSE vs. Target SNR")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename_nse))
    plt.close()

    # 2枚目 (Target SNR vs. Measured SNR)
    filename_snr: str = (
        f"result_snr_vs_snr_"
        f"N{N}_T{T}_"
        f"maxweight{max_weight}_variancee{variance_e}_"
        f"K{K}_Sissymmetric{S_is_symmetric}_"
        f"seed{seed}_"
        f"timestamp{timestamp}.png"
    )
    plt.figure(figsize=(8, 6))
    plt.plot(snr_values, snr_measured_list, marker="o", label="Measured SNR")
    plt.xlabel("Target SNR")
    plt.ylabel("Measured SNR (?)")
    plt.grid(True)
    plt.legend()
    plt.title("Target SNR vs. Measured SNR")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename_snr))
    plt.close()

    # Notebook or .py のバックアップ
    copy_ipynb_path: str = os.path.join(save_path, f"{notebook_filename}_backup_{timestamp}.py")
    shutil.copy(notebook_filename, copy_ipynb_path)
    print(f"Notebook file copied to: {copy_ipynb_path}")
