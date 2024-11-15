import numpy as np

def generate_data(start=0, end=10, num_points=1000):
    """
    Gera valores de x e os respectivos valores de f(x) no intervalo de [0,10] e 1000 pontos.
    """
    x = np.linspace(start, end, num_points)
    y = np.exp(-x / 5) * np.sin(3 * x) + 0.5 * np.sin(x)
    return x, y
