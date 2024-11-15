import numpy as np
import matplotlib.pyplot as plt

def calculate_mse_manual(y_true, y_pred):
    """
    Calculate Mean Squared Error manually.
    """
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += (yt - yp) ** 2
    return error / len(y_true)

def calculate_rmse_manual(y_true, y_pred):
    """
    Calculate Root Mean Squared Error manually.
    """
    mse = calculate_mse_manual(y_true, y_pred)
    return mse ** 0.5

def plot_comparison(x, y_true, y_pred):
    """
    Plot true function and fuzzy approximation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label='True Function', linewidth=2)
    plt.plot(x, y_pred, label='Fuzzy Approximation', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('True Function vs. Fuzzy Approximation')
    plt.legend()
    plt.grid()
    plt.show()

def plot_error(x, y_true, y_pred):
    """
    Plot error between true function and fuzzy approximation.
    """
    error = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.plot(x, error, label='Error', color='red')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Error between True Function and Fuzzy Approximation')
    plt.grid()
    plt.show()
