import numpy as np
from optimization import recursive_least_squares  # Importando o RLS existente

class FuzzySystemTS:
    def __init__(self, num_rules=3, order=1, membership_type='gaussian', operator='min'):
        """
        Inicializa o sistema fuzzy Takagi-Sugeno.
        Args:
            num_rules (int): Número de regras fuzzy.
            order (int): Ordem do modelo Takagi-Sugeno (0 ou 1).
            membership_type (str): Tipo de função de pertinência ('gaussian', 'triangular', 'trapezoidal').
            operator (str): Operador fuzzy para combinar pertinências ('min' ou 'prod').
        """
        self.num_rules = num_rules
        self.order = order
        self.membership_type = membership_type
        self.operator = operator  # Armazena o operador selecionado
        self.membership_functions = []
        self.consequents = np.zeros((num_rules, 2))  # (a, b) para cada regra
        self.P = [np.eye(2) * 1e6 for _ in range(num_rules)]  # Matriz P para cada regra

    def define_membership_functions(self, x_range):
        """
        Define as funções de pertinência para a variável de entrada.
        """
        min_x, max_x = x_range
        step = (max_x - min_x) / (self.num_rules - 1)
        for i in range(self.num_rules):
            center = min_x + i * step
            if self.membership_type == 'gaussian':
                self.membership_functions.append((center, step / 2))
            elif self.membership_type == 'triangular':
                left = max(min_x, center - step)
                right = min(max_x, center + step)
                self.membership_functions.append((left, center, right))
            elif self.membership_type == 'trapezoidal':
                left = max(min_x, center - step)
                right = min(max_x, center + step)
                shoulder_left = max(min_x, left - step / 2)
                shoulder_right = min(max_x, right + step / 2)
                self.membership_functions.append((shoulder_left, left, right, shoulder_right))
    
    def calculate_memberships(self, x):
        """
        Calcula os valores de pertinência para todas as regras e retorna como uma lista.
        """
        memberships = []
        for params in self.membership_functions:
            if self.membership_type == 'gaussian':
                center, width = params
                membership = self.gaussian(x, center, width)
            elif self.membership_type == 'triangular':
                left, center, right = params
                membership = self.triangular(x, left, center, right)
            elif self.membership_type == 'trapezoidal':
                shoulder_left, left, right, shoulder_right = params
                membership = self.trapezoidal(x, shoulder_left, left, right, shoulder_right)
            memberships.append(membership)
        
        return memberships  # Retorna uma lista de pertinências individuais
    
    def gaussian(self, x, center, width):
        """
        Função de pertinência Gaussiana.
        """
        exponent = -((x - center) ** 2) / (2 * width ** 2)
        return self.exp(exponent)
    
    def triangular(self, x, left, center, right):
        """
        Função de pertinência Triangular.
        """
        if x <= left or x >= right:
            return 0
        elif left < x < center:
            return (x - left) / (center - left)
        elif center <= x < right:
            return (right - x) / (right - center)
        return 0
    
    def trapezoidal(self, x, shoulder_left, left, right, shoulder_right):
        """
        Função de pertinência Trapezoidal.
        """
        if x <= shoulder_left or x >= shoulder_right:
            return 0
        elif shoulder_left < x < left:
            return (x - shoulder_left) / (left - shoulder_left)
        elif left <= x <= right:
            return 1
        elif right < x < shoulder_right:
            return (shoulder_right - x) / (shoulder_right - right)
        return 0
    
    def exp(self, value):
        """
        Aproximação manual da função exponencial.
        """
        result = 1
        term = 1
        for i in range(1, 15):
            term *= value / i
            result += term
        return result

    def infer(self, x, desired_output=None):
        """
        Realiza a inferência usando o modelo Takagi-Sugeno e ajusta os consequentes com RLS, se desejado.
        """
        memberships = self.calculate_memberships(x)
        
        # Combinar pertinências usando o operador especificado
        if self.operator == 'prod':
            combined_membership = np.prod(memberships)
        else:  # Default para 'min'
            combined_membership = np.min(memberships)
        
        sum_memberships = sum(memberships)

        # Calcular a saída com consequentes lineares para ordem 1
        if self.order == 1:
            outputs = np.array([a * x + b for a, b in self.consequents])
            output = np.dot(memberships, outputs) / sum_memberships if sum_memberships != 0 else 0
        else:  # Ordem zero
            output = np.dot(memberships, self.consequents[:, 1]) / sum_memberships if sum_memberships != 0 else 0

        # Ajuste com RLS se um valor desejado for fornecido
        if desired_output is not None:
            for i in range(self.num_rules):
                # Preparar o vetor de entrada para o RLS: [x * m_i, m_i]
                x_i = np.array([memberships[i] * x, memberships[i]])
                y_i = desired_output
                # Atualizar os parâmetros (a, b) e a matriz de covariância P para cada regra
                self.consequents[i], self.P[i] = recursive_least_squares(x_i, y_i, self.consequents[i], self.P[i], lambda_reg=0.99)

        return output
