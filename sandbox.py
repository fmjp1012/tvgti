import numpy as np

def soft_threshold(x, thresh):
    """
    ソフトしきい値関数：
      soft_threshold(x, thresh) = sign(x) * max(|x| - thresh, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def proximal_gradient(Y, X, lambda_, step_A, step_B, max_iter=1000, tol=1e-4):
    """
    問題
      min_{A,B} 0.5 * ||Y - A*Y - B*X||_F^2 + lambda * ||A||_1
    subject to diag(A)=0, Bは対角行列
    をプロキシマル勾配法により解く（静的バッチ版）
    
    入力:
      Y         : ノード×キャスケードの観測行列
      X         : ノード×キャスケードの外部影響行列
      lambda_   : ℓ₁正則化パラメータ
      step_A    : A更新用ステップサイズ
      step_B    : B更新用ステップサイズ（対角要素のみ更新）
      max_iter  : 最大反復回数
      tol       : 収束判定の閾値
      
    出力:
      A_est     : 推定されたネットワーク隣接行列（対角は0）
      B_est     : 推定された対角行列（B = diag(b)）
    """
    n, c = Y.shape

    # 初期化: Aはゼロ行列, Bは対角成分を1に初期化（対角のみ扱うのでベクトルで管理）
    A = np.zeros((n, n))
    b = np.ones(n)

    for it in range(max_iter):
        # 対角行列Bを作成
        B = np.diag(b)
        # 残差：Y - A Y - B X
        R = Y - A @ Y - B @ X
        
        # Aに関する勾配： grad_A = - (R Y^T)
        grad_A = - R @ Y.T
        
        # Bに関する勾配：Bは対角行列なので，勾配は対角成分のみ更新
        grad_B_full = - R @ X.T  # shape (n,n)
        grad_b = np.diag(grad_B_full)
        
        # Aの更新（プロキシマル勾配ステップ＋ソフトしきい値）
        A_new = A - step_A * grad_A
        A_new = soft_threshold(A_new, step_A * lambda_)
        # 対角成分はゼロに強制
        np.fill_diagonal(A_new, 0.0)
        
        # Bの更新：対角成分のみ勾配降下
        b_new = b - step_B * grad_b

        # 収束判定（更新前後の変化が閾値以下なら終了）
        if np.linalg.norm(A_new - A, ord='fro') < tol and np.linalg.norm(b_new - b) < tol:
            A, b = A_new, b_new
            break

        A, b = A_new, b_new

    B_est = np.diag(b)
    return A, B_est

# 動作確認用の例（ランダムなデータを用いたシンプルなシミュレーション）
if __name__ == "__main__":
    np.random.seed(0)
    n = 10   # ノード数
    c = 20   # キャスケード（サンプル数）
    # ランダムに観測データ Y と外部影響データ X を生成
    Y = np.random.randn(n, c)
    X = np.random.randn(n, c)
    
    # 正則化パラメータやステップサイズの設定
    lambda_ = 0.1
    step_A = 1e-3
    step_B = 1e-3
    max_iter = 1000
    tol = 1e-4

    A_est, B_est = proximal_gradient(Y, X, lambda_, step_A, step_B, max_iter, tol)

    print("推定されたA（ネットワーク隣接行列、対角はゼロ）:")
    print(A_est)
    print("\n推定されたB（対角行列）:")
    print(B_est)
