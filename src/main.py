from data_generator import generate_data
from fuzzy_system import FuzzySystemTS
from evaluation import calculate_mse, calculate_rmse, plot_comparison, plot_error

def run_test(num_rules, order, membership_type):
    """
    Executa o teste com as configurações especificadas para o sistema fuzzy.
    """
    # Passo 1: Gerar os dados
    x, y_true = generate_data()

    # Passo 2: Inicializar o sistema fuzzy com as configurações desejadas
    fuzzy_system = FuzzySystemTS(num_rules=num_rules, order=order, membership_type=membership_type)
    fuzzy_system.define_membership_functions((min(x), max(x)))

    # Passo 3: Aproximar a função com ajuste dinâmico RLS
    y_pred = [fuzzy_system.infer(xi, desired_output=yi) for xi, yi in zip(x, y_true)]

    # Passo 4: Avaliar o modelo
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    print(f"\nConfiguração: Função de Pertinência = {membership_type}, Número de Regras = {num_rules}, Ordem = {order}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # Passo 5: Visualizar os resultados
    title_config = f"Pertinência = {membership_type}, Regras = {num_rules}, Ordem = {order}"
    plot_comparison(x, y_true, y_pred)
    plot_error(x, y_true, y_pred)


# Executar testes com diferentes configurações
print("Iniciando Testes com Diferentes Configurações de Funções de Pertinência e Número de Regras:")
run_test(num_rules=10, order=1, membership_type='gaussian')    # Função de Pertinência Gaussiana
run_test(num_rules=10, order=1, membership_type='triangular')  # Função de Pertinência Triangular
run_test(num_rules=10, order=1, membership_type='trapezoidal') # Função de Pertinência Trapezoidal
