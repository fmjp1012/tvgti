from scipy.sparse import coo_matrix, save_npz, load_npz
import numpy as np

import os
from functools import wraps

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
