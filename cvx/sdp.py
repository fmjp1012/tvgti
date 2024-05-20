import cvxpy as cvx

def sdp(A):
    n = A.shape[0]
    # 行列用意
    X = cvx.Variable((n, n), symmetric=True)
    # 問題定義
    constraints = [X >> 0]  # 半正定値制約
    obj = cvx.Minimize(cvx.norm(A-X, "fro")**2)
    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True)
    # 結果表示
    return X.value
