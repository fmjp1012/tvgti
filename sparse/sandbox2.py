import numpy as np

def simulate_time_varying_graph_data(
    S: np.ndarray,
    T: int,
    snr_linear: float = None,
    snr_db: float = None,
    random_seed: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate graph signals x_t from the SEM model x_t = (I - S)^{-1} ε_t
    and add measurement noise to achieve a specified SNR.

    Args:
        S (np.ndarray): Adjacency matrix of size (d, d) with zero diagonal.
        T (int): Number of time steps (samples) to simulate.
        snr_linear (float, optional): Desired SNR in linear scale (power ratio).
        snr_db (float, optional): Desired SNR in decibels. If provided, overrides snr_linear.
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        X_noisy (np.ndarray): Observed signals of shape (d, T) with added noise.
        X_true (np.ndarray): True noise-driven signals of shape (d, T) without measurement noise.
        noise (np.ndarray): Measurement noise of shape (d, T).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    d = S.shape[0]
    I = np.eye(d)
    # Compute the SEM transform: (I - S)^{-1}
    M = np.linalg.inv(I - S)

    # Generate true signals x_t = M @ ε_t with ε_t ~ N(0, I)
    eps = np.random.randn(d, T)
    X_true = M @ eps

    # Estimate average signal power per sample
    signal_power = np.mean(np.sum(X_true**2, axis=0))

    # Determine linear SNR
    if snr_db is not None:
        snr = 10 ** (snr_db / 10)
    elif snr_linear is not None:
        snr = snr_linear
    else:
        raise ValueError("Specify either snr_linear or snr_db")

    # Compute measurement noise variance
    noise_variance = signal_power / snr
    noise = np.sqrt(noise_variance) * np.random.randn(d, T)

    # Observed signals
    X_noisy = X_true + noise
    return X_noisy, X_true, noise


if __name__ == "__main__":
    # Example usage
    d = 10
    # Create a random sparse adjacency matrix S with no self-loops
    S = np.random.uniform(-0.1, 0.1, size=(d, d))
    np.fill_diagonal(S, 0)

    T = 1000
    # Simulate with 20 dB SNR
    X_noisy, X_true, noise = simulate_time_varying_graph_data(
        S, T, snr_db=20, random_seed=42
    )

    # Empirical SNR check
    emp_signal_power = np.mean(np.sum(X_true**2, axis=0))
    emp_noise_power = np.mean(np.sum(noise**2, axis=0))
    print(f"Empirical SNR (linear): {emp_signal_power/emp_noise_power:.2f}")
    print(f"Empirical SNR (dB): {10*np.log10(emp_signal_power/emp_noise_power):.2f} dB")
