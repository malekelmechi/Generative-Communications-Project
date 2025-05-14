import json
import matplotlib.pyplot as plt
import os

# \uD83D\uDCC1 Fichier contenant les pertes sauvegardées
loss_file = 'losses/l.json'

# ✅ Vérifier que le fichier existe
if not os.path.exists(loss_file):
    print("❌ Fichier 'losses/l.json' introuvable.")
    exit()

# \uD83D\uDCD6 Charger les données
with open(loss_file, 'r') as f:
    data = json.load(f)

# \uD83D\uDCCA Extraire les époques et les pertes moyennes
epochs = [entry['epoch'] + 1 for entry in data]  # +1 pour commencer à 1
losses = [entry['cross_entropy_loss'] for entry in data]

# \uD83D\uDCC8 Tracer la courbe
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='blue', label='Cross Entropy Loss')
plt.title("Courbe de loss sur les époques")
plt.xlabel("Époque")
plt.ylabel("Loss moyenne")
plt.grid(True)
plt.legend()
plt.tight_layout()

# \uD83D\uDCBE Enregistrer la figure + affichage
plt.savefig("loss_curve.png")
plt.show()