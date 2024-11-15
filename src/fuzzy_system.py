import numpy as np

class FuzzySystemTS:
    def __init__(self, num_rules=3, order=0):
        """
        Initialize Takagi-Sugeno fuzzy system.
        Args:
            num_rules (int): Number of fuzzy rules.
            order (int): Order of the Takagi-Sugeno model (0 or 1).
        """
        self.num_rules = num_rules
        self.order = order
        self.membership_functions = []
        self.rules = []
    
    def define_membership_functions(self, x_range):
        """
        Define membership functions for input variable.
        """
        min_x, max_x = x_range
        step = (max_x - min_x) / (self.num_rules - 1)
        for i in range(self.num_rules):
            center = min_x + i * step
            self.membership_functions.append(
                lambda x, c=center, w=step/2: np.exp(-((x - c) ** 2) / (2 * w ** 2))
            )
    
    def calculate_memberships(self, x):
        """
        Calculate membership values for all rules.
        """
        return np.array([mf(x) for mf in self.membership_functions])
    
    def infer(self, x):
        """
        Perform inference using Takagi-Sugeno model.
        """
        memberships = self.calculate_memberships(x)
        # Simple consequents for order 0 (constant outputs)
        if self.order == 0:
            consequents = np.linspace(-1, 1, self.num_rules)
        else:
            consequents = [lambda x, a=i: a * x for i in range(self.num_rules)]
        
        output = 0
        for i, m in enumerate(memberships):
            output += m * (consequents[i](x) if self.order == 1 else consequents[i])
        return output / np.sum(memberships)
