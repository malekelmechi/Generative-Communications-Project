import json
import matplotlib.pyplot as plt

# Charger les pertes depuis le fichier
with open("losses/losses.json", "r") as f:
    data = json.load(f)

# Extraire les époques et les pertes moyennes
epochs = [entry["epoch"] + 1 for entry in data]
losses = [entry["cross_entropy_loss"] for entry in data]

# Tracer la courbe
plt.plot(epochs, losses, marker='o')
plt.xlabel("Époque")
plt.ylabel("Loss (Cross Entropy)")
plt.title("Courbe de perte pendant l'entraînement")
plt.grid(True)
plt.show()  