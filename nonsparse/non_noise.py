import numpy as np
import cvxpy as cp

def solve_offline_sem(X_up_to_t: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    与えられた信号行列 X_up_to_t に対して、対角成分が 0 の対称行列 S を
    オフラインで求める。目的関数は Frobenius ノルム (Sx ≈ x) の誤差。
    """
    N, t = X_up_to_t.shape
    S = cp.Variable((N, N), symmetric=True)
    
    # 正則化項付きの目的関数（今回は lambda_reg=0 としている）
    # objective = (1/(2*t)) * cp.norm(X_up_to_t - S @ X_up_to_t, 'fro') + lambda_reg * cp.norm1(S)
    objective = (1/(2*t)) * cp.norm(X_up_to_t - S @ X_up_to_t, 'fro')
    
    constraints = [cp.diag(S) == 0]
    
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    prob.solve(solver=cp.SCS, verbose=False)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("CVXPY did not find an optimal solution.")
    
    S_opt = S.value
    return S_opt

def compute_ground_truth_S(x: np.ndarray) -> np.ndarray:
    """
    ベクトル x に対して、対称かつ対角成分0の行列 S を Sx = x となるように構成する。
    ただし、自由度がある中で「最小ノルム解」を求めるため、変数は
    i < j の S_ij (全 n(n-1)/2 個) とみなし、線形系 A s = x を解く。
    """
    N = x.shape[0]
    m = N*(N-1)//2  # 自由変数の数
    A = np.zeros((N, m))
    index_pairs = []  # どの変数がどの行列成分に対応するか記録
    col = 0
    for i in range(N):
        for j in range(i+1, N):
            index_pairs.append((i, j))
            # 行 i では係数は x_j, 行 j では係数は x_i
            A[i, col] = x[j]
            A[j, col] = x[i]
            col += 1
    # A s = x を最小二乗（最小ノルム）解で求める：s = A^T (A A^T)^{-1} x
    B = A @ A.T  # サイズ N x N
    y = np.linalg.solve(B, x)
    s = A.T @ y  # 自由変数ベクトル（長さ m）
    
    # s の値を用いて S (対称行列・対角成分0) を構成する
    S_true = np.zeros((N, N))
    for k, (i, j) in enumerate(index_pairs):
        S_true[i, j] = s[k]
        S_true[j, i] = s[k]
    return S_true

def run_simulation(N=5, T=100, seed=0):
    """
    シミュレーションのメイン関数
      - N 次元のベクトル x をランダムに生成（各成分が非零となるように）
      - x の T 個のコピーからなる信号行列 X を作成
      - 正解の S (最小ノルム解) を compute_ground_truth_S() で構成
      - solve_offline_sem() で推定した S を得る
      - Sx = x となっているか（残差）および、正解 S との違いを評価
    """
    np.random.seed(seed)
    # 非零成分となるようにランダム生成（ゼロに近すぎる場合は再生成）
    x = np.random.randn(N)
    while np.any(np.abs(x) < 1e-2):
        x = np.random.randn(N)
        
    # 信号 X は x の T 個のコピー（各列が x）とする：サイズ (N, T)
    X = np.tile(x, (T, 1)).T

    # 正解の S を求める
    S_true = compute_ground_truth_S(x)
    
    # オフライン推定により S を求める
    S_est = solve_offline_sem(X, lambda_reg=0.0)
    
    # S_est が Sx = x を満たすかどうかの残差を計算
    residual = np.linalg.norm(S_est @ x - x)
    # 正解 S と推定 S の Frobenius ノルムの差
    diff = np.linalg.norm(S_est - S_true)
    
    print("生成したベクトル x:")
    print(x)
    print("\n[正解] 最小ノルム解 S (対称・対角成分0, Sx=x を満たす):")
    print(S_true)
    print("\nオフライン推定により得られた S:")
    print(S_est)
    print("\n残差 ||S_est @ x - x|| =", residual)
    print("正解 S と推定 S の Frobenius ノルムの差 =", diff)

if __name__ == "__main__":
    run_simulation()
