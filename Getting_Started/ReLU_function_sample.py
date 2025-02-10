import numpy as np
import matplotlib.pyplot as plt

# Define ReLU function
def relu(x):
    return np.maximum(0, x)

# Generate values from -10 to 10
x = np.linspace(-10, 10, 100)
y = relu(x)

# Plot ReLU
plt.plot(x, y, label="ReLU Function", color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("Input (x)")
plt.ylabel("Output (f(x))")
plt.title("ReLU Activation Function")
plt.legend()
plt.grid()
plt.show()
