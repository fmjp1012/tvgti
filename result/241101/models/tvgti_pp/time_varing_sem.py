import numpy as np
from tqdm import tqdm
from utils import elimination_matrix_hh, duplication_matrix_hh, project_to_zero_diagonal_symmetric

class TimeVaryingSEM:
    def __init__(self, N, lambda_reg, alpha, beta, gamma, P, C):
        """
        Time-Varying Structural Equation Model (TV-SEM) Prediction-Correction Algorithm

        Parameters:
        - N: Number of nodes (variables) in the graph
        - lambda_reg: Regularization parameter for L1 norm
        - alpha: Step size for prediction step
        - beta: Step size for correction step
        - gamma: Forgetting factor for covariance update
        - P: Number of prediction iterations
        - C: Number of correction iterations
        """
        self.N = N
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.P = P
        self.C = C

        # Vectorization: hollow half-vectorization
        self.l = N * (N - 1) // 2
        self.D_h = duplication_matrix_hh(N).tocsc()
        self.D_h_T = self.D_h.T
        self.E_h = elimination_matrix_hh(N).tocsc()
        self.E_h_T = self.E_h.T
        
        # Initialize the graph shift operator as a hollow symmetric matrix
        self.S = np.zeros((N, N))
        # Initialize the vectorized S
        self.s = self.E_h @ self.S.flatten()

    def correction_step(self, x):
        """
        Performs the correction step with C iterations.

        Parameters:
        - s_pred: Predicted s from the prediction step

        Returns:
        - s_corr: Corrected s after C iterations
        """
        self.S = self.S - self.beta * 2 * (self.S @ x @ x.T - x @ x.T)
        self.S = project_to_zero_diagonal_symmetric(self.S)
        self.s = self.E_h @ self.S.flatten()
    
    def run(self, X):
        """
        Runs the TV-SEM algorithm on the provided data stream.

        Parameters:
        - X: Iterable of data vectors of shape (N,)

        Returns:
        - estimates: List of estimated S matrices over time
        - errors: List of estimation errors (optional)
        """
        estimates = []
        for t, x in enumerate(tqdm(X.T)):
            # Correction step
            self.correction_step(x)
            
            # Reconstruct the symmetric hollow S matrix
            S_flat = self.D_h @ self.s
            S_matrix = np.zeros((self.N, self.N))
            idx = 0
            for i in range(self.N):
                for j in range(i+1, self.N):
                    S_matrix[i, j] = self.s[idx]
                    S_matrix[j, i] = self.s[idx]
                    idx += 1
            self.S = S_matrix
            
            estimates.append(self.S.copy())
        
        return estimates
