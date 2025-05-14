import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils import initNetParams, train_step
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader

# 🔧 Paramètres
CHECKPOINT_PATH = 'checkpoints/deepsc-NoChannel'
RESUME_EPOCH = 149
TOTAL_EPOCHS = RESUME_EPOCH + 50  # = 199
LR = 0.001
BATCH_SIZE = 128
VOCAB_FILE = 'data/vocab.json'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 📖 Charger le vocabulaire
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
token_to_idx = vocab.get("token_to_idx", vocab)
num_vocab = len(token_to_idx)
pad_idx = token_to_idx.get("<PAD>", 0)

# 🧠 Charger le modèle
max_len = 32

deepsc = DeepSC(4, num_vocab, num_vocab, max_len, max_len, 128, 8, 512, 0.1).to(device)
ckpt_path = os.path.join(CHECKPOINT_PATH, f"checkpoint_epoch_{RESUME_EPOCH}.pth")
deepsc.load_state_dict(torch.load(ckpt_path, map_location=device))  # 🔧 modifié ici
print(f"✅ Poids chargés depuis : {ckpt_path}")


# ⚙ Optimiseur + Critère
optimizer = torch.optim.Adam(deepsc.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)

# 📦 Dataset
train_dataset = EurDataset('train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0,
                          pin_memory=True, collate_fn=collate_data)

# 🔁 Boucle d'entraînement
def train(epoch):
    deepsc.train()
    pbar = tqdm(train_loader)
    losses = []
    for sents in pbar:
        sents = sents.to(device)
        input_ids = sents[:, :-1]
        target_ids = sents[:, 1:]
        loss = train_step(deepsc, input_ids, target_ids, pad_idx, optimizer, criterion)
        losses.append(loss.item())
        pbar.set_description(f"Epoch {epoch + 1} | LR: {LR:.5f} | CE Loss: {loss:.4f}")
    avg = np.mean(losses)
    print(f" Moyenne loss: {avg:.4f}")
    return avg

# 📊 Sauvegarde des pertes
def save_loss(epoch, avg_loss):
    os.makedirs("losses", exist_ok=True)
    loss_file = 'losses/l.json'  # ⬅️ changé ici
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            all_losses = json.load(f)
    else:
        all_losses = []
    all_losses.append({"epoch": epoch, "cross_entropy_loss": float(avg_loss)})
    with open(loss_file, 'w') as f:
        json.dump(all_losses, f, indent=4)

# 🚀 Entraînement continuation
for epoch in range(RESUME_EPOCH, TOTAL_EPOCHS):
    print(f" Reprise Epoch {epoch + 1}/{TOTAL_EPOCHS} | LR = {LR}")
    avg_loss = train(epoch)
    # 💾 Checkpoint
    ckpt = os.path.join(CHECKPOINT_PATH, f"checkpoint_epoch_{epoch + 1}.pth")
    torch.save(deepsc.state_dict(), ckpt)
    print(f"✅ Checkpoint sauvegardé : {ckpt}")
    save_loss(epoch, avg_loss)
