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
        self.consequents = []
    
    def define_membership_functions(self, x_range):
        """
        Define membership functions for input variable.
        """
        min_x, max_x = x_range
        step = (max_x - min_x) / (self.num_rules - 1)
        for i in range(self.num_rules):
            center = min_x + i * step
            self.membership_functions.append((center, step / 2))
    
    def calculate_memberships(self, x):
        """
        Calculate membership values for all rules.
        """
        memberships = []
        for center, width in self.membership_functions:
            # Gaussian-like membership function
            membership = self.gaussian(x, center, width)
            memberships.append(membership)
        return memberships
    
    def gaussian(self, x, center, width):
        """
        Gaussian membership function without numpy.
        """
        exponent = -((x - center) ** 2) / (2 * width ** 2)
        return self.exp(exponent)
    
    def exp(self, value):
        """
        Approximate exponential function manually.
        """
        result = 1
        term = 1
        for i in range(1, 15):  # Approximation using 15 terms of series
            term *= value / i
            result += term
        return result

    def infer(self, x):
        """
        Perform inference using Takagi-Sugeno model.
        """
        memberships = self.calculate_memberships(x)
        sum_memberships = sum(memberships)
        
        if self.order == 0:
            # Simple consequents for order 0
            self.consequents = [(i / self.num_rules) for i in range(self.num_rules)]
            output = sum(m * c for m, c in zip(memberships, self.consequents))
        else:
            # Consequents for order 1 (linear functions)
            self.consequents = [(lambda x, a=i: a * x) for i in range(self.num_rules)]
            output = sum(m * c(x) for m, c in zip(memberships, self.consequents))
        
        return output / sum_memberships
