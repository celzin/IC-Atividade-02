import numpy as np

def recursive_least_squares(x_i, y_i, theta, P, lambda_reg=0.99):
    """
    Atualiza os parâmetros usando Recursive Least Squares (RLS) para uma única entrada.
    
    Args:
        x_i (numpy.ndarray): Vetor de entrada para a amostra atual.
        y_i (float): Valor alvo para a amostra atual.
        theta (numpy.ndarray): Vetor de parâmetros atuais (coeficientes a serem ajustados).
        P (numpy.ndarray): Matriz de covariância.
        lambda_reg (float): Fator de esquecimento (0 < lambda_reg <= 1).
    
    Returns:
        tuple: Vetor de parâmetros atualizado e matriz de covariância atualizada.
    """
    # Converter x_i para um vetor coluna
    x_i = x_i.reshape(-1, 1)
    
    # Calcular o ganho de Kalman
    K = P @ x_i / (lambda_reg + x_i.T @ P @ x_i)
    
    # Atualizar os parâmetros com base no erro atual
    error = y_i - (theta @ x_i).item()
    theta = theta + (K.flatten() * error)
    
    # Atualizar a matriz de covariância
    P = (P - K @ x_i.T @ P) / lambda_reg
    
    return theta, P
