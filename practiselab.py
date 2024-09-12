import numpy as np
import matplotlib.pyplot as plt

# Define the range of input values
x = np.linspace(-10, 10, 1000)

# Step Function
def step_function(x):
    return np.where(x >= 0, 1, 0)

def step_derivative(x):
    return np.zeros_like(x)

# Linear Function
def linear_function(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Tanh Function
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# ReLU Function
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Leaky ReLU Function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# ELU Function
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, elu(x, alpha) + alpha)

# Softplus Function
def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_derivative(x):
    return 1 / (1 + np.exp(-x))

# Swish Function
def swish(x, beta=1):
    return x * sigmoid(beta * x)

def swish_derivative(x, beta=1):
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)

# Softsign Function
def softsign(x):
    return x / (1 + np.abs(x))

def softsign_derivative(x):
    return 1 / (1 + np.abs(x)) ** 2

# Plotting the functions and their derivatives
activation_functions = [
    ("Step", step_function, step_derivative),
    ("Linear", linear_function, linear_derivative),
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("Tanh", tanh, tanh_derivative),
    ("ReLU", relu, relu_derivative),
    ("Leaky ReLU", leaky_relu, leaky_relu_derivative),
    ("ELU", elu, elu_derivative),
    ("Softplus", softplus, softplus_derivative),
    ("Swish", swish, swish_derivative),
    ("Softsign", softsign, softsign_derivative)
]

fig, axes = plt.subplots(len(activation_functions), 2, figsize=(12, 40))

for i, (name, func, deriv) in enumerate(activation_functions):
    axes[i, 0].plot(x, func(x))
    axes[i, 0].set_title(f"{name} Function")
    axes[i, 1].plot(x, deriv(x))
    axes[i, 1].set_title(f"{name} Derivative")

plt.tight_layout()
plt.show()