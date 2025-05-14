import torch

print("CUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nom du GPU :", torch.cuda.get_device_name(0))
else:
    print("❌ Le GPU n'est pas détecté. On tourne sur CPU.")
