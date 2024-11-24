import numpy as np
import cvxpy as cp
from scipy.linalg import norm
from tqdm import tqdm
from utils import project_to_zero_diagonal_symmetric

class TimeVaryingSEM:
    def __init__(self, N, S_0, r, q, rho):
        self.N = N

        self.l = N * (N - 1) // 2
        self.S = S_0

        self.r = r # window size
        self.q = q # number of processors

        self.rho = rho
        self.w = 1 / q

    def g_l(self, x): # per processor
        return norm(x - self.S @ x) ** 2 / 2 - self.rho

    def subgrad_projection(self, x):
        if self.g_l(x) > 0:
            subgrad = self.S @ x @ x.T - x @ x.T
            return self.S - (self.g_l(x) / (norm(subgrad) ** 2)) * subgrad 
        else:
            return self.S
    
    def parallel_projection(self, X_partial):
        in_all_C_l = True
        sum_weighted_projection_sp = np.zeros((self.N, self.N))

        numerator = 0.0
        denominator = 0.0


        for i in range(self.q):
            x_per_processor = X_partial[:, i: self.r]

            projection_sp = self.subgrad_projection(x_per_processor)

            sum_weighted_projection_sp += self.w * projection_sp

            numerator += self.w * norm(projection_sp - self.S) ** 2

            if self.g_l(x_per_processor) > 0:
                in_all_C_l = False

        # print("in_all_C_l: " + str(in_all_C_l))
        # print("sum_weighted_projection_sp: " + str(norm(sum_weighted_projection_sp)))
        if not in_all_C_l:
            assert numerator > 0
            denominator = norm(sum_weighted_projection_sp - self.S) ** 2
            M_k = numerator / denominator
            # print("M_k: " + str(M_k))

            self.S = self.S + M_k * (sum_weighted_projection_sp - self.S)
            np.fill_diagonal(self.S, 0)  # Ensure diagonal elements are zero
        else:
            M_k = 1
            self.S = self.S + M_k * (sum_weighted_projection_sp - self.S)

    def run(self, X):
        estimates = []
        for t, x in enumerate(tqdm(X.T, desc="pp_nonsparse")):
            # print("start------------------------------")
            # print("t: " + str(t))
            if t - self.q - self.r + 2 >= 0:
                self.parallel_projection(X[:, t - self.q - self.r + 2: t+1])
            
            # print("norm of S: " + str(norm(self.S)))
            # print("end------------------------------")
            estimates.append(self.S.copy())
        
        return estimates
