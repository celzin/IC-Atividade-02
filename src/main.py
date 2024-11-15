import numpy as np
from data_generator import generate_data
from fuzzy_system import FuzzySystemTS
from evaluation import calculate_mse, calculate_rmse, plot_comparison, plot_error

# Step 1: Generate Data
x, y_true = generate_data(num_points=1000)

# Step 2: Initialize Fuzzy System
fuzzy_system = FuzzySystemTS(num_rules=5, order=0)
fuzzy_system.define_membership_functions((x.min(), x.max()))

# Step 3: Approximate Function
y_pred = np.array([fuzzy_system.infer(xi) for xi in x])

# Step 4: Evaluate the Model
mse = calculate_mse(y_true, y_pred)
rmse = calculate_rmse(y_true, y_pred)
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# Step 5: Plot Results
plot_comparison(x, y_true, y_pred)
plot_error(x, y_true, y_pred)
