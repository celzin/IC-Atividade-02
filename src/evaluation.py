import numpy as np
import matplotlib.pyplot as plt

def calculate_mse_manual(y_true, y_pred):
    """
    Calcula o Erro Médio Quadrático (MSE) manualmente.
    """
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += (yt - yp) ** 2
    return error / len(y_true)

def calculate_rmse_manual(y_true, y_pred):
    """
    Calcula a Raiz do Erro Médio Quadrático (RMSE) manualmente.
    """
    mse = calculate_mse_manual(y_true, y_pred)
    return mse ** 0.5

def plot_comparison(x, y_true, y_pred):
    """
    Plota a função real e a aproximação fuzzy.
    """
    for i in range(0, len(x), len(x) // 10):
        print(f"x = {x[i]:.2f}, Real: {y_true[i]:.4f}, Aproximado: {y_pred[i]:.4f}")

def plot_error(x, y_true, y_pred):
    """
    Mostra os erros calculados entre o valor real e aproximado.
    """
    error = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.plot(x, error, label='Error', color='red')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Error between True Function and Fuzzy Approximation')
    plt.grid()
    plt.show()
