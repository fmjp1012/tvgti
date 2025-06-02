import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from scipy.optimize import bisect

# 再現性のためにseedを設定
np.random.seed(42)

def compute_snr(c, S_rand, d):
    """
    入力:
      c      : スケーリング定数
      S_rand : 対称かつ対角成分が0のランダムな d×d 行列
      d      : 次元
    出力:
      S = c * S_rand を用いたときの SNR
    """
    # スケーリングされた S の生成
    S = c * S_rand
    # I - S の計算
    I_minus_S = np.eye(d) - S
    # 逆行列の計算
    inv_I_minus_S = np.linalg.inv(I_minus_S)
    # SNR = (1/d) * trace[(I-S)^{-1}(I-S)^{-T}]
    snr = np.trace(inv_I_minus_S @ inv_I_minus_S.T) / d
    return snr

# --- 設定 ---
gamma_target = 10   # 目標 SNR (100->10に変更)
d = 3                # 次元 (5->3に変更)

# --- 1. 対称なランダム行列 S_rand の生成 ---
# まず、d×d の一様乱数行列 A を生成
A = np.random.uniform(low=-1.0, high=1.0, size=(d, d))
# 対称行列にするため、A とその転置の平均を取る
S_rand = (A + A.T) / 2
# 対角成分は必ず0にする
np.fill_diagonal(S_rand, 0)
print("生成した対称な S_rand 行列:\n", S_rand)

# --- 2. 安定性のための c の上限の計算 ---
# S_rand の固有値（スペクトル半径）を計算
eigvals = np.linalg.eigvals(S_rand)
spectral_radius = np.max(np.abs(eigvals))
print("S_rand のスペクトル半径 =", spectral_radius)

# 安定性条件: |c| < 1 / spectral_radius
# ここでは gamma_target > 1 を仮定して c > 0 として探索します
# c_max = 1.0 / spectral_radius - 1e-6  # 1e-6 で余裕を持たせる
c_max = 1.0 / spectral_radius - 1e-6  # 1e-6 で余裕を持たせる

# --- 3. 二分法による c の探索 ---
# f(c) = compute_snr(c, S_rand, d) - gamma_target
# c = 0 のとき SNR=1 なので f(0)=1-gamma_target (<0 となるはず)
f0 = compute_snr(0, S_rand, d) - gamma_target
f_max = compute_snr(c_max, S_rand, d) - gamma_target
print("f(0) =", f0, " f(c_max) =", f_max)

# c の解を [0, c_max] 内で探索
c_sol = bisect(lambda c: compute_snr(c, S_rand, d) - gamma_target, 0, c_max)
print("求めたスケーリング定数 c =", c_sol)
print("この c のときの SNR =", compute_snr(c_sol, S_rand, d))

# --- 4. 最終的な対称な S 行列の生成 ---
S_final = c_sol * S_rand
print("生成した最終的な対称な S 行列:\n", S_final)

eigvals = np.linalg.eigvals(S_final)
spectral_radius = np.max(np.abs(eigvals))
print("S_final のスペクトル半径 =", spectral_radius)