import os
from functools import wraps

import numpy as np
import cvxpy as cp
from scipy.sparse import coo_matrix, save_npz, load_npz

def persistent_cache(cache_dir="matrix_cache"):
    """
    ディスク上に行列をキャッシュするデコレータ。

    Parameters:
    - cache_dir (str): キャッシュファイルを保存するディレクトリのパス。

    Returns:
    - decorator (function): デコレータ関数。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(N):
            # キャッシュディレクトリの作成
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            # キャッシュファイル名の生成
            cache_filename = f"{func.__name__}_N={N}.npz"
            cache_path = os.path.join(cache_dir, cache_filename)
            
            # キャッシュの存在確認
            if os.path.isfile(cache_path):
                try:
                    # キャッシュファイルの読み込み
                    matrix = load_npz(cache_path)
                    # print(f"Loaded cached matrix from {cache_path}")
                    return matrix
                except Exception as e:
                    print(f"Failed to load cache from {cache_path}: {e}")
                    # キャッシュが破損している場合は再計算
            # キャッシュが存在しない場合、関数を実行してキャッシュを保存
            matrix = func(N)
            try:
                save_npz(cache_path, matrix)
                print(f"Saved matrix to cache at {cache_path}")
            except Exception as e:
                print(f"Failed to save cache to {cache_path}: {e}")
            return matrix
        return wrapper
    return decorator

@persistent_cache()
def elimination_matrix_h(N):
    """
    Generates the elimination matrix E for h-space (half-vectorization).

    Parameters:
    - N (int): The dimension of the square matrix S (N x N).

    Returns:
    - E (scipy.sparse.coo_matrix): The elimination matrix of size (k x N^2),
      where k = N(N + 1)/2.
    """
    k = N * (N + 1) // 2
    rows = []
    cols = []
    data = []
    count = 0
    for j in range(N):
        for i in range(j + 1):
            rows.append(count)
            cols.append(i * N + j)
            data.append(1)
            count += 1
    E = coo_matrix((data, (rows, cols)), shape=(k, N * N))
    return E

@persistent_cache()
def duplication_matrix_h(N):
    """
    Generates the duplication matrix D for h-space (half-vectorization).

    Parameters:
    - N (int): The dimension of the square matrix S (N x N).

    Returns:
    - D (scipy.sparse.coo_matrix): The duplication matrix of size (N^2 x k),
      where k = N(N + 1)/2.
    """
    k = N * (N + 1) // 2
    rows = []
    cols = []
    data = []
    for j in range(N):
        for i in range(j + 1):
            vech_index = j * (j + 1) // 2 + i
            # Assign to (i,j)
            rows.append(i * N + j)
            cols.append(vech_index)
            data.append(1)
            if i != j:
                # Assign to (j,i) as well
                rows.append(j * N + i)
                cols.append(vech_index)
                data.append(1)
    D = coo_matrix((data, (rows, cols)), shape=(N * N, k))
    return D

@persistent_cache()
def elimination_matrix_hh(N):
    """
    Generates the elimination matrix E_h for hh-space (hollow half-vectorization).

    Parameters:
    - N (int): The dimension of the square matrix S (N x N).

    Returns:
    - E_h (scipy.sparse.coo_matrix): The elimination matrix of size (k' x N^2),
      where k' = N(N - 1)/2.
    """
    k_prime = N * (N - 1) // 2
    rows = []
    cols = []
    data = []
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            rows.append(count)
            cols.append(i * N + j)
            data.append(1)
            count += 1
    E_h = coo_matrix((data, (rows, cols)), shape=(k_prime, N*N))
    return E_h

@persistent_cache()
def duplication_matrix_hh(N):
    """
    Generates the duplication matrix D_h for hh-space (hollow half-vectorization).

    Parameters:
    - N (int): The dimension of the square matrix S (N x N).

    Returns:
    - D_h (scipy.sparse.coo_matrix): The duplication matrix of size (N^2 x k'),
      where k' = N(N - 1)/2.
    """
    k_prime = N * (N - 1) // 2
    rows = []
    cols = []
    data = []
    for i in range(N):
        for j in range(i+1, N):
            vechh_index = i * (N - 1) - (i*(i-1))//2 + j - i - 1
            # Assign to (i,j)
            rows.append(i * N + j)
            cols.append(vechh_index)
            data.append(1)
            # Assign to (j,i) as well
            rows.append(j * N + i)
            cols.append(vechh_index)
            data.append(1)
    D_h = coo_matrix((data, (rows, cols)), shape=(N*N, k_prime))
    return D_h

def soft_thresholding( x, threshold):
    """
    Applies the soft-thresholding operator element-wise.

    Parameters:
    - x: Input vector
    - threshold: Threshold value

    Returns:
    - Proximal operator result
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

def firm_shrinkage(x, t1, t2):
    """
    Firm Shrinkageを適用します。

    パラメータ:
    x : array-like
        入力データ。
    t1 : float
        下限のしきい値（0より大きい）。
    t2 : float
        上限のしきい値（t1より大きい）。

    戻り値:
    numpy.ndarray
        しきい値処理後のデータ。
    """
    x = np.asarray(x)
    abs_x = np.abs(x)
    y = np.zeros_like(x)
    sign_x = np.sign(x)
    
    # 場合1: |x| <= t1
    mask1 = abs_x <= t1
    y[mask1] = 0
    
    # 場合2: t1 < |x| <= t2
    mask2 = (abs_x > t1) & (abs_x <= t2)
    y[mask2] = (t2 * (abs_x[mask2] - t1) / (t2 - t1)) * sign_x[mask2]
    
    # 場合3: |x| > t2
    mask3 = abs_x > t2
    y[mask3] = x[mask3]
    
    return y

def project_to_zero_diagonal_symmetric(matrix):
    # 行列を対称化
    symmetric_matrix = (matrix + matrix.T) / 2
    # 対角成分を0に設定
    np.fill_diagonal(symmetric_matrix, 0)
    return symmetric_matrix

def solve_offline_sem(X_up_to_t: np.ndarray, lambda_reg: float) -> np.ndarray:
    N, t = X_up_to_t.shape
    S = cp.Variable((N, N), symmetric=True)
    
    # objective = (1/(2*t)) * cp.norm(X_up_to_t - S @ X_up_to_t, 'fro') + lambda_reg * cp.norm1(S)
    objective = (1/(2*t)) * cp.norm(X_up_to_t - S @ X_up_to_t, 'fro')
    
    constraints = [cp.diag(S) == 0]
    
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    prob.solve(solver=cp.SCS, verbose=False)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("CVXPY did not find an optimal solution.")
    
    S_opt = S.value
    return S_opt

def calc_snr(S: np.ndarray) -> float:
    """
    与えられた行列 S (NxN) に対して、
    SNR = (1/N) * tr( (I - S)^-1 * (I - S)^-T ) を計算する。
    """
    N = S.shape[0]
    I = np.eye(N)
    inv_mat = np.linalg.inv(I - S)  # (I - S)^-1
    # (I - S)^-1 (I - S)^-T = inv_mat @ inv_mat.T
    val = np.trace(inv_mat @ inv_mat.T)
    return val / N

def scale_S_for_target_snr(S: np.ndarray, snr_target: float,
                           tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """
    与えられた S (NxN) をスケーリングする係数 alpha を見つけて、
    (I - alpha*S) が可逆 & スペクトル半径 < 1 となる範囲で
    目標とする snr_target に近い SNR を実現するように返す。
    """
    # 前提: S は正方行列
    N = S.shape[0]
    
    # スペクトル半径を計算
    eigvals = np.linalg.eigvals(S)
    rho_S = max(abs(eigvals))
    
    # もし rho_S == 0 なら、S=0 の場合などで SNR=1 が常に得られる
    # ここでは簡単に場合分け
    if rho_S == 0:
        current_snr = calc_snr(S * 0.0)  # = 1/N * tr(I * I^T) = 1
        if abs(current_snr - snr_target) < tol:
            return S  # そのまま
        else:
            # どうにもならないので、とりあえず返しておく
            return S
    
    # alpha の上限: ここでは 1/(rho_S + ちょっとのマージン) とする
    alpha_high = 1.0 / rho_S * 0.999  # 安全のため少しだけ小さめにする
    alpha_low = 0.0
    
    # 2分探索
    for _ in range(max_iter):
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        
        # (I - alpha*S) が可逆かチェック -> np.linalg.inv がエラーを吐かないか確かめる
        try:
            tmp_snr = calc_snr(alpha_mid * S)
        except np.linalg.LinAlgError:
            # 可逆でなかったら、もう少し alpha を小さくする
            alpha_high = alpha_mid
            continue
        
        if tmp_snr > snr_target:
            # 目標より SNR が高いので、alpha を小さく
            alpha_high = alpha_mid
        else:
            # 目標より SNR が低いので、alpha を大きく
            alpha_low = alpha_mid
        
        # 収束チェック
        if abs(tmp_snr - snr_target) < tol:
            break
    
    alpha_star = 0.5 * (alpha_low + alpha_high)
    return alpha_star * S