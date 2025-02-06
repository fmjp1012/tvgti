import numpy as np
import scipy.linalg as la
from scipy.optimize import root_scalar

def compute_SNR(S, sigma=1):
    """
    与えられた S に対して，
    信号生成モデル x = (I - S)^{-1} ε, ε ~ N(0, sigma^2 I)
    の SNR = (1/d)*tr((I-S)^{-1}(I-S)^{-T}) を計算する．
    （ここでは sigma の影響は打ち消されるため sigma=1 としています．）
    """
    d = S.shape[0]
    I = np.eye(d)
    A = la.inv(I - S)
    return np.trace(A @ A.T) / d

def generate_S_for_SNR(target_SNR, d, method='scalar', random_state=None):
    """
    target_SNR: 目標とする SNR 値 (target_SNR>=1 が望ましい)
    d: ベクトルの次元
    method: 'scalar' あるいは 'random'
       - 'scalar': S = c I として c = 1 - 1/sqrt(target_SNR) とする．
       - 'random': 対角成分が 0 のランダム行列 R を生成し，
                   S = α R として f(α)=SNR(α)-target_SNR = 0 を数値解法で解く．
    random_state: 乱数の種 (任意)
    
    戻り値:
       目標 SNR に対応する S 行列
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if method == 'scalar':
        # S = c I として SNR = 1/(1-c)^2 となるので
        c = 1 - 1/np.sqrt(target_SNR)
        S = c * np.eye(d)
        return S

    elif method == 'random':
        # 対角成分ゼロのランダム行列 R を生成（例: 一様分布 [-1,1]）
        R = np.random.uniform(low=-1, high=1, size=(d, d))
        np.fill_diagonal(R, 0)

        # α を変数として f(α)=SNR(α)-target_SNR を定義
        def f(alpha):
            I = np.eye(d)
            try:
                A = la.inv(I - alpha * R)
            except la.LinAlgError:
                return np.inf  # 非正則の場合は大きな値を返す
            current_SNR = np.trace(A @ A.T) / d
            return current_SNR - target_SNR

        # α の取りうる上限：I - α R の正則性を保つためには
        # α < 1/ρ(R) となるので，まず 1/∥R∥₂ を上限の目安とする．
        sigma_max = la.norm(R, 2)
        if sigma_max == 0:
            if np.isclose(target_SNR, 1.0):
                return R  # R=0 なら S=0 で SNR=1
            else:
                raise ValueError("Rが零行列のため目標SNRに合わせられません．")
        alpha_max = 0.99 / sigma_max  # 安全率をかける
        
        # f(0) = 1 - target_SNR （target_SNR>=1なら f(0)<=0）
        # α を大きくしたとき f(α) が正になる区間があると仮定して，ブランケットを求める．
        f_alpha_max = f(alpha_max)
        while f_alpha_max < 0:
            alpha_max *= 1.1
            f_alpha_max = f(alpha_max)
            if alpha_max > 10 / sigma_max:  # ループ脱出用
                raise ValueError("適切な α の上限が見つかりませんでした．")
        
        # ブレシクション法で f(α)=0 を解く
        sol = root_scalar(f, bracket=[0, alpha_max], method='bisect')
        if not sol.converged:
            raise RuntimeError("α の解が収束しませんでした．")
        alpha_opt = sol.root

        S = alpha_opt * R
        return S
    else:
        raise ValueError("method は 'scalar' か 'random' を指定してください．")

# --- 使用例 ---
if __name__ == "__main__":
    target_SNR = 10.0  # 例：目標 SNR を2に設定
    d = 5             # 次元を5とする

    # 方法1: scalar method
    S_scalar = generate_S_for_SNR(target_SNR, d, method='scalar')
    snr_scalar = compute_SNR(S_scalar)
    
    # 方法2: random method (乱数の種を固定)
    S_random = generate_S_for_SNR(target_SNR, d, method='random', random_state=42)
    snr_random = compute_SNR(S_random)
    
    print("【scalar method】")
    print("生成された S 行列:")
    print(S_scalar)
    print("計算された SNR:", snr_scalar)
    
    print("\n【random method】")
    print("生成された S 行列:")
    print(S_random)
    print("計算された SNR:", snr_random)
