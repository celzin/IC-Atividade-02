import numpy as np
from data_generator import generate_data
from fuzzy_system import FuzzySystemTS
from evaluation import calculate_mse, calculate_rmse, plot_comparison, plot_error

# Etapa 1: Gerar os Dados com 1000 pontos
x, y_true = generate_data(num_points=1000)

# Etapa 2: Inicializar o Sistema Fuzzy
fuzzy_system = FuzzySystemTS(num_rules=5, order=0)
fuzzy_system.define_membership_functions((x.min(), x.max()))

# Etapa 3: Aproximar a Função
y_pred = np.array([fuzzy_system.infer(xi) for xi in x])

# Etapa 4: Avaliar o Modelo
mse = calculate_mse(y_true, y_pred)
rmse = calculate_rmse(y_true, y_pred)
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# Etapa 5: Plotar Resultados
plot_comparison(x, y_true, y_pred)
plot_error(x, y_true, y_pred)
