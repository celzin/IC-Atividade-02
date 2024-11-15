import numpy as np

def recursive_least_squares(X, Y, lambda_reg=0.99):
    """
    Recursive Least Squares (RLS) optimization.
    Args:
        X: Input features (matrix).
        Y: Target values (vector).
        lambda_reg: Forgetting factor (0 < lambda_reg <= 1).
    Returns:
        Parameters (weights) for the linear system.
    """
    P = np.eye(X.shape[1]) * 1e6
    theta = np.zeros(X.shape[1])
    
    for i in range(X.shape[0]):
        x_i = X[i, :].reshape(-1, 1)
        y_i = Y[i]
        K = P @ x_i / (lambda_reg + x_i.T @ P @ x_i)
        theta += (K.flatten() * (y_i - theta @ x_i.flatten()))
        P = (P - K @ x_i.T @ P) / lambda_reg
    
    return theta
