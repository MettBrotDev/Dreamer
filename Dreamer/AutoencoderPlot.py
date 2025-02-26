import re
import matplotlib.pyplot as plt

log_filename = "world_model.log"

steps = []
losses = []

with open(log_filename, "r") as f:
    for line in f:
        line = line.strip()
        step, loss = line.split(",")
        step = int(step)
        loss = float(loss)
        steps.append(step)
        losses.append(loss)

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label="Loss", color="blue", marker="o", markersize=3, linestyle="-")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss vs. Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_filename = log_filename.strip(".txt").strip(".log") + ".png"
plt.savefig(output_filename)
plt.show()