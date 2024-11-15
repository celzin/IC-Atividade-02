class FuzzySystemTS:
    def __init__(self, num_rules=3, order=0):
        """
        Inicializa o sistema fuzzy Takagi-Sugeno.
        Args:
            num_rules (int): Número de regras fuzzy.
            order (int): Ordem do modelo Takagi-Sugeno (0 ou 1).
        """
        self.num_rules = num_rules
        self.order = order
        self.membership_functions = []
        self.consequents = []
    
    def define_membership_functions(self, x_range):
        """
        Define as funções de pertinência para a variável de entrada.
        """
        min_x, max_x = x_range
        step = (max_x - min_x) / (self.num_rules - 1)
        for i in range(self.num_rules):
            center = min_x + i * step
            self.membership_functions.append((center, step / 2))
    
    def calculate_memberships(self, x):
        """
        Calcula os valores de pertinência para todas as regras.
        """
        memberships = []
        for center, width in self.membership_functions:
            membership = self.gaussian(x, center, width)
            memberships.append(membership)
        return memberships
    
    def gaussian(self, x, center, width):
        """
        Função de pertinência Gaussiana sem uso de bibliotecas.
        """
        exponent = -((x - center) ** 2) / (2 * width ** 2)
        return self.exp(exponent)
    
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

    def infer(self, x):
        """
        Realiza a inferência usando o modelo Takagi-Sugeno.
        """
        memberships = self.calculate_memberships(x)
        sum_memberships = sum(memberships)
        if self.order == 0:
            # Consequentes simples para ordem 0
            self.consequents = [(i / self.num_rules) for i in range(self.num_rules)]
            output = sum(m * c for m, c in zip(memberships, self.consequents))
        else:
            # Consequentes para ordem 1 (funções lineares)
            self.consequents = [(lambda x, a=i: a * x) for i in range(self.num_rules)]
            output = sum(m * c(x) for m, c in zip(memberships, self.consequents))
        return output / sum_memberships
