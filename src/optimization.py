import numpy as np

def recursive_least_squares(X, Y, lambda_reg=0.99):
    """
    Otimização usando Recursive Least Squares (RLS).
    
    Args:
        X (numpy.ndarray): Matriz de características de entrada.
        Y (numpy.ndarray): Vetor de valores-alvo.
        lambda_reg (float): Fator de esquecimento (0 < lambda_reg <= 1).
    
    Returns:
        numpy.ndarray: Vetor de parâmetros (pesos) ajustados.
    """
    # Inicializa a matriz de covariância P e o vetor de parâmetros theta
    P = np.eye(X.shape[1]) * 1e6
    theta = np.zeros(X.shape[1])
    
    for i in range(X.shape[0]):
        # Extrai o vetor de entrada atual
        x_i = X[i, :].reshape(-1, 1)
        # Valor alvo correspondente
        y_i = Y[i]
        
        # Calcula o ganho de Kalman
        K = P @ x_i / (lambda_reg + x_i.T @ P @ x_i)
        
        # Atualiza os parâmetros com base no erro atual
        theta += (K.flatten() * (y_i - theta @ x_i.flatten()))
        
        # Atualiza a matriz de covariância
        P = (P - K @ x_i.T @ P) / lambda_reg
    
    return theta
