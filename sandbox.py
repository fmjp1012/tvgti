import numpy as np
import matplotlib.pyplot as plt

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
    """
    n, c = Y.shape

    # 初期化: Aはゼロ行列, Bは対角成分を1に初期化（対角のみ扱うのでベクトルで管理）
    A = np.zeros((n, n))
    b = np.ones(n)

    for it in range(max_iter):
        # 対角行列Bを作成
        B = np.diag(b)
        # 残差：Y - A*Y - B*X
        R = Y - A @ Y - B @ X
        
        # Aに関する勾配： grad_A = - (R * Y^T)
        grad_A = - R @ Y.T
        
        # Bに関する勾配：Bは対角行列なので，勾配は対角成分のみ更新
        grad_B_full = - R @ X.T  # shape (n, n)
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

# --- 以下、真のパラメータでデータ生成・評価・ヒートマップ表示 ---

if __name__ == "__main__":
    np.random.seed(0)
    n = 10   # ノード数
    c = 20   # サンプル数（キャスケード数）

    # 真のA_trueを生成：対角成分は0，オフダイアゴナルは確率pでランダム値（小さい値）を持つ
    p = 0.3  # スパース性の割合
    A_true = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and np.random.rand() < p:
                # 値を -0.5～0.5 の一様分布から取得
                A_true[i, j] = np.random.uniform(-0.5, 0.5)
    # 安定性確保のため全体をスケーリング（スペクトル半径 < 1となるように）
    A_true *= 0.2

    # 真のB_trueを生成：各ノードの対角成分（例：0.5～1.5の一様乱数）
    b_true = np.random.uniform(0.5, 1.5, size=n)
    B_true = np.diag(b_true)

    # 外部影響 X を生成
    X = np.random.randn(n, c)
    # 真のモデルに従い、Y を生成
    # Y = (I - A_true)^{-1} B_true X　※(I - A_true)が非特異であることを仮定
    I = np.eye(n)
    Y = np.linalg.inv(I - A_true) @ (B_true @ X)

    # 推定用のパラメータ
    lambda_ = 0.1
    step_A = 1e-3
    step_B = 1e-3
    max_iter = 1000
    tol = 1e-4

    # 推定実行
    A_est, B_est = proximal_gradient(Y, X, lambda_, step_A, step_B, max_iter, tol)

    # 推定結果と真の行列との誤差を計算（Frobeniusノルム）
    err_A = np.linalg.norm(A_est - A_true, ord='fro')
    err_B = np.linalg.norm(B_est - B_true, ord='fro')
    print("推定されたAと真のAのFrobenius距離:", err_A)
    print("推定されたBと真のBのFrobenius距離:", err_B)

    # ヒートマップ表示
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im0 = axes[0, 0].imshow(A_true, cmap='viridis')
    axes[0, 0].set_title("True A")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(A_est, cmap='viridis')
    axes[0, 1].set_title("Estimated A")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(B_true, cmap='viridis')
    axes[1, 0].set_title("True B")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(B_est, cmap='viridis')
    axes[1, 1].set_title("Estimated B")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()
