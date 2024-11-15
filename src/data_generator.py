def generate_data(start=0, end=10, num_points=1000):
    """
    Gera valores de x e os respectivos valores de f(x).
    """
    step = (end - start) / (num_points - 1)
    x = [start + i * step for i in range(num_points)]
    y = [exp(-xi / 5) * sin(3 * xi) + 0.5 * sin(xi) for xi in x]
    return x, y

def exp(value):
    """
    Calcula a função exponencial manualmente usando série de Taylor.
    """
    result = 1
    term = 1
    for i in range(1, 15):
        term *= value / i
        result += term
    return result

def sin(value):
    """
    Calcula a função seno manualmente usando série de Taylor.
    """
    result = 0
    term = value
    for i in range(15):
        sign = -1 if i % 2 else 1
        result += sign * term
        term *= value ** 2 / ((2 * i + 2) * (2 * i + 3))
    return result