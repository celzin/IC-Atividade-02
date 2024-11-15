import matplotlib.pyplot as plt

def calculate_mse(y_true, y_pred):
    """
    Calcula o Erro Médio Quadrático (MSE) manualmente.
    """
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += (yt - yp) ** 2
    return error / len(y_true)

def calculate_rmse(y_true, y_pred):
    """
    Calcula a Raiz do Erro Médio Quadrático (RMSE) manualmente.
    """
    mse = calculate_mse(y_true, y_pred)
    return mse ** 0.5

def plot_comparison(x, y_true, y_pred, title_config):
    """
    Plota a função real e a aproximação fuzzy com título detalhado.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label='Função Real', linewidth=2)
    plt.plot(x, y_pred, label='Aproximação Fuzzy', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Função Real vs. Aproximação Fuzzy\n{title_config}')
    plt.legend()
    plt.grid()
    plt.show()

def plot_error(x, y_true, y_pred, title_config):
    """
    Plota o erro entre a função real e a aproximação fuzzy com título detalhado.
    """
    error = [yt - yp for yt, yp in zip(y_true, y_pred)]
    plt.figure(figsize=(10, 6))
    plt.plot(x, error, label='Erro', color='red')
    plt.xlabel('x')
    plt.ylabel('Erro')
    plt.title(f'Erro entre a Função Real e a Aproximação Fuzzy\n{title_config}')
    plt.grid()
    plt.show()
