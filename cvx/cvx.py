import numpy as np
import cvxpy as cvx

np.random.seed(2)
# 次元数
N = 3
T = 20000
X = np.random.rand(N, N)
Theta_true = X @ X.T
S_true = np.linalg.inv(Theta_true)
print("Theta_true----------------------------")
print(Theta_true)
print("--------------------------------------")
print("S_true--------------------------------")
print(S_true)
print("--------------------------------------")
data = np.random.multivariate_normal(mean=np.zeros(N), cov=Theta_true, size=T)
# 各種定数・変数
S = cvx.Variable((N, N), symmetric=True)  # 変数定義
# 問題設定
# サンプル共分散行列
sample_cov = np.cov(data, rowvar=False)
# 目的関数
obj = cvx.Minimize(-cvx.log_det(S) + cvx.trace(S @ sample_cov))  # 最小化
# 制約
constraint = [S >> 0]  # 制約
# 問題
prob = cvx.Problem(obj, constraint)  # 問題
# 解く
prob.solve(verbose=True)  # 解く
# 表示
print("obj: ", prob.value)
print("S: ", S.value)
# フロベニウスノルムでの距離を計算
frobenius_norm_diff = np.linalg.norm(S.value - S_true, 'fro')
frobenius_norm_true = np.linalg.norm(S_true, 'fro')
nse = (frobenius_norm_diff ** 2) / (frobenius_norm_true ** 2)
print("NSE between S and S_true: ", nse)
