from data_generator import generate_data
from fuzzy_system import FuzzySystemTS
from evaluation import calculate_mse, calculate_rmse, plot_comparison, plot_error

# Passo 1: Gerar os dados
x, y_true = generate_data()

# Passo 2: Inicializar o sistema fuzzy
fuzzy_system = FuzzySystemTS(num_rules=5, order=0)
fuzzy_system.define_membership_functions((min(x), max(x)))

# Passo 3: Aproximar a função
y_pred = [fuzzy_system.infer(xi) for xi in x]

# Passo 4: Avaliar o modelo
mse = calculate_mse(y_true, y_pred)
rmse = calculate_rmse(y_true, y_pred)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Passo 5: Visualizar os resultados
plot_comparison(x, y_true, y_pred)
plot_error(x, y_true, y_pred)
