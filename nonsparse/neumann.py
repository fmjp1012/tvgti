import numpy as np
import matplotlib.pyplot as plt

def neumann_series_inverse(S, num_terms):
    """
    Neumann series inverse matrix approximation
    (I-S)^{-1} ≈ I + S + S^2 + ... + S^(num_terms-1)
    """
    d = S.shape[0]
    I = np.eye(d)
    series_sum = np.copy(I)
    current_term = np.copy(I)
    for k in range(1, num_terms):
        current_term = S @ current_term
        series_sum += current_term
    return series_sum

def simulate_neumann():
    np.random.seed(0)
    d = 5  # Matrix dimension

    # 1. Generate random symmetric matrix R (diagonal=0)
    R = np.random.randn(d, d)
    R = (R + R.T) / 2  # Symmetrize
    np.fill_diagonal(R, 0)  # Zero diagonal

    # 2. Calculate spectral radius
    eigenvalues = np.linalg.eigvals(R)
    spectral_radius = np.max(np.abs(eigenvalues))
    print("Spectral radius of R:", spectral_radius)

    I = np.eye(d)
    
    # 3. Convergent case: c < 1/spectral_radius
    c_conv = 0.8 / spectral_radius
    S_conv = c_conv * R
    inv_direct_conv = np.linalg.inv(I - S_conv)

    # 4. Calculate errors for convergent case
    num_terms_list = np.arange(1, 51)
    errors_conv = []
    for n in num_terms_list:
        inv_neumann_conv = neumann_series_inverse(S_conv, n)
        error = np.linalg.norm(inv_neumann_conv - inv_direct_conv, ord='fro')
        errors_conv.append(error)

    # 5. Divergent case: c > 1/spectral_radius
    c_div = 1.2 / spectral_radius
    S_div = c_div * R
    inv_direct_div = np.linalg.inv(I - S_div)

    errors_div = []
    for n in num_terms_list:
        inv_neumann_div = neumann_series_inverse(S_div, n)
        error = np.linalg.norm(inv_neumann_div - inv_direct_div, ord='fro')
        errors_div.append(error)

    # 6. Plot results in English
    plt.figure(figsize=(10, 5))
    plt.plot(num_terms_list, errors_conv, 'o-', label=f"Convergent: c = {c_conv:.3f}")
    plt.plot(num_terms_list, errors_div, 's-', label=f"Divergent: c = {c_div:.3f}")
    plt.xlabel("Number of Terms in Neumann Series")
    plt.ylabel("Error (Frobenius Norm)")
    plt.yscale("log")
    plt.title("Neumann Series Matrix Inverse Approximation Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate_neumann()