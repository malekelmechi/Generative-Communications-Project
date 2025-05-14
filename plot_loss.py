import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# Chemin du fichier de pertes
loss_file = "losses/losses.json"

# Vérifier que le fichier existe
if not os.path.exists(loss_file):
    print("❌ Fichier 'losses.json' introuvable.")
    exit()

# Charger les données brutes
with open(loss_file, "r") as f:
    raw_data = json.load(f)

# Supprimer les doublons : garder la dernière perte pour chaque epoch
unique_losses = {}
for entry in raw_data:
    epoch = entry["epoch"]
    loss = entry["cross_entropy_loss"]
    unique_losses[epoch] = loss

# Trier les époques
epochs = sorted(unique_losses.keys())
losses = [unique_losses[ep] for ep in epochs]

# Option : version lissée (filtre de Gauss)
smooth_losses = gaussian_filter1d(losses, sigma=2)

# Tracer la courbe
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label="Loss brute", marker='o', alpha=0.4)
plt.plot(epochs, smooth_losses, label="Loss lissée", color='blue', linewidth=2)
plt.xlabel("Époque")
plt.ylabel("Perte (Cross Entropy)")
plt.title("Courbe de perte pendant l'entraînement")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()
