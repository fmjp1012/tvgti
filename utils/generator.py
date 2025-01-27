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

def generate_piecewise_X_K(N: int, T: int, S_is_symmetric: bool, sparsity: float,
                           max_weight: float, std_e: float, K: int) -> Tuple[List[np.ndarray], np.ndarray]:
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

def generate_brownian_piecewise_X_K(
    N: int,
    T: int,
    S_is_symmetric: bool,
    sparsity: float,
    max_weight: float,
    std_e: float,
    K: int,
    std_S: float = 0.1  # ← 追加: S をランダムに「揺らす」強度
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
    S = generate_random_S(N, sparsity=sparsity, max_weight=max_weight)
    
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