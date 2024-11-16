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
    x_i = np.atleast_2d(x_i).T 

    # Calcular o ganho de Kalman
    Px = P @ x_i  
    denom = lambda_reg + x_i.T @ Px  
    K = Px / denom  

    # Calcular o erro de predição
    error = y_i - (x_i.T @ theta) 

    # Atualizar os parâmetros
    theta = theta + K.flatten() * error 

    # Atualizar a matriz de covariância
    P = (P - K @ x_i.T @ P) / lambda_reg 

    return theta, P
