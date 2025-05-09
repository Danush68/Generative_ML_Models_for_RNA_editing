import pandas as pd
import matplotlib.pyplot as plt

# Load the training loss log
loss_log = pd.read_csv("../src/training_loss_log.csv")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(loss_log["Epoch"], loss_log["Loss"], marker='o', linestyle='-')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
