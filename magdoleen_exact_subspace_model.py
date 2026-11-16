import matplotlib.pyplot as plt
import numpy as np


class MagdoleenExactSubspaceModel:
  """Model Name: Magdoleen Exact Subspace Functional Model (MESFM)

  Author: Dr. Magdoleen Mutseam Mudawi
  """

  def __init__(self, lambda_ratio=0.25, p_fractal=2.5, n_iterations=400):
    self.lambda_ratio = lambda_ratio
    self.p = p_fractal
    self.n_iterations = n_iterations
    self.weights = None
    self.cost_history = []

  def _fractal_thresholding(self, z, threshold):
    abs_z = np.abs(z)
    mask = abs_z > threshold
    adaptive_shift = threshold / (1.0 + (abs_z / (threshold + 1e-8)) ** self.p)
    return np.where(mask, np.sign(z) * (abs_z - adaptive_shift), 0.0)

  def fit(self, X, y):
    m, n = X.shape
    self.weights = np.zeros(n)
    self.cost_history = []

    # Lipschitz constant (Spectral Norm)
    L = np.linalg.norm(X, ord=2) ** 2
    alpha = 1.0 / L

    # Stage 1: Fractal Functional Thresholding
    for i in range(self.n_iterations):
      residual = X.dot(self.weights) - y
      gradient = X.T.dot(residual)
      w_temp = self.weights - alpha * gradient

      current_thresh = self.lambda_ratio * alpha * np.max(np.abs(X.T.dot(y)))
      self.weights = self._fractal_thresholding(w_temp, current_thresh)

      mse_cost = 0.5 * np.linalg.norm(residual) ** 2
      self.cost_history.append(mse_cost)

    # Stage 2: Hilbert Subspace Orthogonal Projection
    max_peak = np.max(np.abs(self.weights))
    if max_peak > 0:
      active_support = np.where(np.abs(self.weights) > 0.20 * max_peak)[0]
      if len(active_support) > 0:
        X_subspace = X[:, active_support]
        w_exact, _, _, _ = np.linalg.lstsq(X_subspace, y, rcond=None)
        self.weights = np.zeros(n)
        self.weights[active_support] = w_exact

    return self

  def predict(self, X):
    return X.dot(self.weights)


if __name__ == "__main__":
  np.random.seed(42)
  m_measurements = 40
  n_features = 100

  X = np.random.randn(m_measurements, n_features) / np.sqrt(m_measurements)
  true_weights = np.zeros(n_features)
  true_weights[[10, 25, 50, 75, 90]] = [2.5, -1.8, 3.2, -2.0, 1.5]

  noise = np.random.normal(0, 0.05, size=m_measurements)
  y = X.dot(true_weights) + noise

  model = MagdoleenExactSubspaceModel()
  model.fit(X, y)

  error = np.linalg.norm(model.weights - true_weights) / np.linalg.norm(
      true_weights
  )
  print(f"Model successfully executed! Recovery Relative Error: {error:.2%}")
