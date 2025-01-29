from typing import List, Tuple, Dict

import numpy as np
from scipy.linalg import inv, eigvals, norm

from utils import *

def generate_random_S(
    N: int,
    sparsity: float,
    max_weight: float,
    S_is_symmetric: bool
) -> np.ndarray:
    """
    対称行列バージョンのサンプル実装.
    sparsity (0~1) に従って要素を0にするかどうかをランダムに決定し，
    [-max_weight, max_weight] の範囲で乱数を生成し対称化したものを返す.
    """
    S = np.random.uniform(-max_weight, max_weight, size=(N, N))
    mask = (np.random.rand(N, N) < (1.0 - sparsity))
    S = S * mask  # スパース化

    # 対称化
    if S_is_symmetric:
        S = (S + S.T) / 2

    spectral_radius = max(abs(eigvals(S)))
    if spectral_radius >= 1:
        S = S / (spectral_radius + 0.1)
    S = S / norm(S)
    return S

def generate_piecewise_X_K(
        N: int,
        T: int,
        S_is_symmetric: bool,
        sparsity: float,
        max_weight: float,
        std_e: float,
        K: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    S_list = []
    inv_I_S_list = []
    I = np.eye(N)
    for i in range(K):
        S = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
        S_list.append(S)
        inv_I_S_list.append(inv(I - S))
    # Divide T into K segments
    segment_lengths = [T // K] * K
    segment_lengths[-1] += T % K
    # Create S_series
    S_series = []
    for i, length in enumerate(segment_lengths):
        S_series.extend([S_list[i]] * length)
    # Generate error terms
    e_t_series = np.random.normal(0, std_e, size=(N, T))
    # Compute X
    X_list = []
    start = 0
    for i, length in enumerate(segment_lengths):
        end = start + length
        X_i = inv_I_S_list[i] @ e_t_series[:, start:end]
        X_list.append(X_i)
        start = end
    X = np.concatenate(X_list, axis=1)
    return S_series, X

def update_S(
    S_prev: np.ndarray,
    noise_std: float,
    S_is_symmetric: bool,
    max_weight: float
) -> np.ndarray:
    """
    前の S にノイズを加えて新しい S を生成するサンプル関数．
    - noise_std: ノイズの標準偏差
    - S_is_symmetric: True の場合は最終的に対称行列にする
    - max_weight: 行列要素の絶対値を制限したい場合に利用
    """
    N = S_prev.shape[0]
    
    # ガウスノイズを加える
    noise = np.random.normal(0, noise_std, size=(N, N))
    S_new = S_prev + noise
    
    # 対称行列にしたい場合は対称化
    if S_is_symmetric:
        S_new = (S_new + S_new.T) / 2
    
    # 要素を [-max_weight, max_weight] 以内にクリップしておく (任意)
    # （S の安定性などを考慮するなら、ここでスペクトル半径を抑える処理も可）
    S_new = np.clip(S_new, -max_weight, max_weight)
    
    return S_new

def generate_piecewise_X_K_with_snr(
    N: int,
    T: int,
    S_is_symmetric: bool,
    sparsity: float,
    max_weight: float,
    std_e: float,
    K: int,
    snr_target: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[List[np.ndarray], np.ndarray]:
    S_list = []
    inv_I_S_list = []
    I = np.eye(N)
    
    # 1) Generate K random S and scale each to snr_target
    for i in range(K):
        # Generate a base random S
        S_raw = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
        # Scale it to achieve the target SNR
        S_scaled = scale_S_for_target_snr(S_raw, snr_target, tol=tol, max_iter=max_iter)
        S_list.append(S_scaled)
        inv_I_S_list.append(np.linalg.inv(I - S_scaled))

    # 2) Divide T into K segments
    segment_lengths = [T // K] * K
    segment_lengths[-1] += T % K  # handle any remainder in the last segment

    # 3) Create S_series (length T)
    S_series = []
    for i, length in enumerate(segment_lengths):
        S_series.extend([S_list[i]] * length)

    # 4) Generate exogenous shock e(t)
    e_t_series = np.random.normal(0, std_e, size=(N, T))

    # 5) Compute X piecewise
    X_list = []
    start = 0
    for i, length in enumerate(segment_lengths):
        end = start + length
        X_i = inv_I_S_list[i] @ e_t_series[:, start:end]
        X_list.append(X_i)
        start = end
    
    X = np.concatenate(X_list, axis=1)
    
    return S_series, X

def generate_brownian_piecewise_X_K(
    N: int,
    T: int,
    S_is_symmetric: bool,
    sparsity: float,
    max_weight: float,
    std_e: float,
    K: int,
    std_S: float  # ← 追加: S をランダムに「揺らす」強度
):
    """
    K 回に分けて S を生成し、それぞれの区間で X を生成する関数.
    今回は S を「全くのランダム」ではなく，前の S に
    ガウスノイズを加える形でランダム運動させる.
    """
    
    # ============== S の生成 ==============
    S_list = []
    inv_I_S_list = []
    I = np.eye(N)
    
    # まず最初の S は従来どおりランダムに生成
    S = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    
    S_list.append(S)
    inv_I_S_list.append(inv(I - S))
    
    # 2回目以降は，前の S にノイズを加える形で生成
    for i in range(1, K):
        S_prev = S_list[-1]
        S_new = update_S(
            S_prev, 
            noise_std=std_S,
            S_is_symmetric=S_is_symmetric,
            max_weight=max_weight
        )
        S_list.append(S_new)
        inv_I_S_list.append(inv(I - S_new))
    
    # ============== 時系列 T を K 区間に分割 ==============
    segment_lengths = [T // K] * K
    segment_lengths[-1] += T % K  # 端数を最後の区間に足す
    
    # ============== S_series (時系列ごとの S ) ==============
    S_series = []
    for i, length in enumerate(segment_lengths):
        S_series.extend([S_list[i]] * length)
    
    # ============== e (外生ショック) の生成 ==============
    e_t_series = np.random.normal(0, std_e, size=(N, T))
    
    # ============== X の計算 ==============
    X_list = []
    start = 0
    for i, length in enumerate(segment_lengths):
        end = start + length
        X_i = inv_I_S_list[i] @ e_t_series[:, start:end]
        X_list.append(X_i)
        start = end
    
    X = np.concatenate(X_list, axis=1)
    
    return S_series, X

def generate_linear_X(
    N: int,
    T: int,
    S_is_symmetric: bool,
    sparsity: float,
    max_weight: float,
    std_e: float
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    - 最初にランダム行列 S_start と S_end を生成し，
    - t=0 から t=T-1 まで線形に補間した S(t) を用いて，
      X(t) = (I - S(t))^-1 * e(t)
    でサンプルを生成するメモリレスSEMの関数．

    返り値:
      S_series: t=0 から t=T-1 までの S(t) を格納したリスト (長さ T)
      X       : 観測データ (N x T)
    """
    I = np.eye(N)

    # 1. 2つの行列をランダム生成
    S_start = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_end   = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    
    # 2. スペクトル半径をチェック → 1未満に抑える
    def spectral_radius(A: np.ndarray) -> float:
        return max(abs(eigvals(A)))
    
    rho_start = spectral_radius(S_start)
    rho_end   = spectral_radius(S_end)
    rho_max   = max(rho_start, rho_end)
    
    # もしスペクトル半径が1を超えているなら0.99に押さえ込む
    if rho_max >= 1.0:
        alpha = 0.99 / rho_max
        S_start *= alpha
        S_end   *= alpha
    
    # 念のため再度チェック (必ず < 1 になっているはず)
    # ここでさらに厳密に <1 であるか確認したい場合は適宜 assert 等しても良い
    # print("rho_start =", spectral_radius(S_start))
    # print("rho_end   =", spectral_radius(S_end))

    # 3. 時刻 t=0..T-1 で線形補間した S(t) をリストに格納
    S_series = []
    inv_I_S_list = []
    for t in range(T):
        if T == 1:
            lam = 0.0
        else:
            lam = t / (T - 1)  # 0 から 1 まで
        
        # 線形補間
        S_t = (1.0 - lam) * S_start + lam * S_end
        
        # ここでも「スペクトル半径 < 1」を保障したい場合は，
        # S_start, S_end がともに <1 なら凸結合 S(t) も <1 になります
        # （スペクトル半径はサブアディティブなので）
        
        S_series.append(S_t)
        inv_I_S_list.append(np.linalg.inv(I - S_t))

    # 4. 外生ショック e(t) の生成
    e_t_series = np.random.normal(0, std_e, size=(N, T))
    
    # 5. X(t) = (I - S(t))^-1 e(t) をまとめて生成
    X_list = []
    for t in range(T):
        x_t = inv_I_S_list[t] @ e_t_series[:, t]
        X_list.append(x_t.reshape(N, 1))

    # shape を (N, T) に整形
    X = np.concatenate(X_list, axis=1)
    
    return S_series, X
