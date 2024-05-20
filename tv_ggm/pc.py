import time
import numpy as np
import matplotlib.pyplot as plt

from utils import *

class OnlineGraphLearning:
   def __init__(self, N, T, P, C, alpha, beta, gamma, S_true):
       self.N = N
       self.T = T
       self.P = P
       self.C = C
       self.alpha = alpha
       self.beta = beta
       self.gamma = gamma
       self.S_true = S_true
       self.total_iterations = T
       self.Theta_hat = None
       self.s_hat = None
       self.Theta_hat_prev = None
       self.nses = []
       self.D = duplication_matrix(self.N)
   
   def predict(self, s_hat, Theta_hat, Theta_hat_prev, alpha, h, D):
       s = s_hat.copy()
       S_inv = np.linalg.inv(vech_to_mat(s_hat))
       grad_f = D.T @ mat_to_vec(Theta_hat - S_inv)
       grad_ts_f = D.T @ mat_to_vec(Theta_hat - Theta_hat_prev)
       s -= 2 * alpha * (grad_f + h * grad_ts_f)
       return self.project(s)
   
   def correct(self, s_hat, Theta_hat, beta, D):
       s = s_hat.copy()
       S_inv = np.linalg.inv(vech_to_mat(s_hat))
       grad_f = D.T @ mat_to_vec(Theta_hat - S_inv)
       s -= beta * grad_f
       return self.project(s)
   
   def project(self, s):
       S = vech_to_mat(s)
       eigvals, eigvecs = np.linalg.eigh(S)
       eigvals = np.maximum(eigvals, 0.01)
       S = eigvecs @ np.diag(eigvals) @ eigvecs.T
       return mat_to_vech(S)
   
   def run(self, data):
       start_time = time.time()
       
       self.Theta_hat = np.cov(data[:10, :], rowvar=False)
       self.s_hat = mat_to_vech(np.linalg.inv(self.Theta_hat))
       self.Theta_hat_prev = self.Theta_hat
       
       for t in range(self.total_iterations):
           if t % (self.total_iterations // 10) == 0:
               elapsed_time = time.time() - start_time
               print(f"Progress: {t / self.total_iterations * 100:3.0f}% ({t:5}/{self.total_iterations:5}), Elapsed time: {elapsed_time:5.2f}s")
           
           for _ in range(self.P):
               self.s_hat = self.predict(self.s_hat, self.Theta_hat, self.Theta_hat_prev, self.alpha, 1, self.D)
           
           x_t = data[t, :]
           self.Theta_hat_prev = self.Theta_hat
           self.Theta_hat = self.gamma * self.Theta_hat_prev + (1 - self.gamma) * np.outer(x_t, x_t)
           
           for _ in range(self.C):
               self.s_hat = self.correct(self.s_hat, self.Theta_hat, self.beta, self.D)
           
           S_hat = vech_to_mat(self.s_hat)
           nse = np.linalg.norm(S_hat - self.S_true, ord='fro') ** 2 / np.linalg.norm(self.S_true, ord='fro') ** 2
           self.nses.append(nse)
       
       elapsed_time = time.time() - start_time
       print(f"Progress: 100% ({self.total_iterations:5}/{self.total_iterations:5}), Elapsed time: {elapsed_time:5.2f}s")
       
       return vech_to_mat(self.s_hat), self.nses

def generate_data(N, T, seed=24):
   np.random.seed(seed)
   X = np.random.rand(N, N)
   Theta_true = np.cov(X)
   eigvals, eigvecs = np.linalg.eigh(Theta_true)
   Theta_true = eigvecs @ np.diag([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) @ eigvecs.T
   S_true = np.linalg.inv(Theta_true)
   data = np.random.multivariate_normal(mean=np.zeros(N), cov=Theta_true, size=T)
   return data, S_true

if __name__ == "__main__":
   N = 10
   T = 20000
   P = 1
   C = 1
   alpha = 0.001
   beta = 0.001
   gamma = 0.999
   seed = np.random.randint(10000)
   print(f"seed: {seed}")
   
   data, S_true = generate_data(N, T, seed)
   print(f'S\'s condition number: {spectral_range(S_true)}')
   
   ogl = OnlineGraphLearning(N, T, P, C, alpha, beta, gamma, S_true)
   S_hat, nses = ogl.run(data)
   
   plt.figure(figsize=(8, 6))
   plt.semilogy(nses)
   plt.xlabel('Iteration')
   plt.ylabel('NSE (log scale)')
   plt.title('Online Graph Learning Convergence')
   plt.grid(True)
   plt.show()