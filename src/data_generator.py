import numpy as np

def generate_data(start=0, end=10, num_points=1000):
    """
    Generate x values and corresponding f(x) values.
    """
    x = np.linspace(start, end, num_points)
    y = np.exp(-x / 5) * np.sin(3 * x) + 0.5 * np.sin(x)
    return x, y
