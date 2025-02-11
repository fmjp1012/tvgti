import numpy as np
from scipy.optimize import bisect

def compute_snr(c, S_rand, d):
    """
    S = c * S_rand を用いたときの SNR を計算する関数
    SNR = (1/d) * trace[(I-S)^{-1}(I-S)^{-T}]
    """
    S = c * S_rand
    I_minus_S = np.eye(d) - S
    inv_I_minus_S = np.linalg.inv(I_minus_S)
    snr = np.trace(inv_I_minus_S @ inv_I_minus_S.T) / d
    return snr

# --- 設定 ---
gamma_target = 100   # 目標 SNR (例)
d = 5                # 次元

# --- 1. 対称なランダム行列 S_rand の生成 ---
# d×d の一様乱数行列 A を生成
A = np.random.uniform(low=-1.0, high=1.0, size=(d, d))
# 対称行列にするため、A とその転置の平均を取る
S_rand = (A + A.T) / 2
# 対角成分は必ず 0 にする
np.fill_diagonal(S_rand, 0)
print("生成した対称な S_rand 行列:\n", S_rand)

# --- 2. f(c) = compute_snr(c, S_rand, d) - gamma_target の定義 ---
def f(c):
    return compute_snr(c, S_rand, d) - gamma_target

# --- 3. c の解を含む区間の自動探索（安定性条件は無視） ---
c_low = 0.0
c_high = 0.1  # 初期の上限値

# f(c_low)= compute_snr(0, ...)-gamma_target ですが、通常は 1 - gamma_target と小さい値になる想定です
while f(c_high) < 0:
    c_high *= 2
    print("拡大中: c_high =", c_high, " f(c_high) =", f(c_high))

print("探索区間: [{}, {}]".format(c_low, c_high))

# --- 4. 二分法による c の探索 ---
c_sol = bisect(f, c_low, c_high)
print("求めたスケーリング定数 c =", c_sol)
print("この c のときの SNR =", compute_snr(c_sol, S_rand, d))

# --- 5. 最終的な対称な S 行列の生成 ---
S_final = c_sol * S_rand
print("生成した最終的な対称な S 行列:\n", S_final)
