import os
import argparse
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import initNetParams, train_step
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-NoChannel', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("✅ CUDA disponible :", torch.cuda.is_available())
print("✅ Utilisation du device :", device)
if torch.cuda.is_available():
    print("✅ Nom du GPU :", torch.cuda.get_device_name(0))

# Fixer les seeds pour reproductibilité
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Supprimer les anciens checkpoints et losses
def clean_previous_runs():
    if os.path.exists("losses"):
        shutil.rmtree("losses")
        print("🧹 Dossier 'losses' supprimé.")
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
        print("🧹 Dossier 'checkpoints' supprimé.")

# Sauvegarder la loss moyenne
def save_losses(cross_entropy_losses, epoch):
    os.makedirs("losses", exist_ok=True)
    loss_file = 'losses/losses.json'
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            all_losses = json.load(f)
    else:
        all_losses = []
    avg_loss = float(np.mean(cross_entropy_losses))
    all_losses.append({
        "epoch": epoch,
        "cross_entropy_loss": avg_loss
    })
    with open(loss_file, 'w') as f:
        json.dump(all_losses, f, indent=4)
    print(f"💾 Loss (avg) sauvegardée pour epoch {epoch}: {avg_loss:.4f}")

# Entraînement principal
def train(epoch, args, net):
    net.train()
    train_eur = EurDataset('train')
    print(f"📦 Train dataset chargé avec {len(train_eur)} échantillons")
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator, desc=f'[Epoch {epoch+1}] Device: {device}')
    cross_entropy_losses = []
    for sents in pbar:
        sents = sents.to(device)
        input_ids = sents[:, :-1]
        target_ids = sents[:, 1:]
        ce_loss = train_step(net, input_ids, target_ids, pad_idx, optimizer, criterion)
        pbar.set_postfix({
            'CE Loss': f"{ce_loss:.5f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.5f}"
        })
        cross_entropy_losses.append(ce_loss.item())
    save_losses(cross_entropy_losses, epoch)

# Point d'entrée principal
if __name__ == '__main__':
    args = parser.parse_args()

    # Nettoyage
    clean_previous_runs()

    # Charger vocabulaire
    vocab_path = os.path.join(os.path.dirname(__file__), args.vocab_file)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    token_to_idx = vocab.get("token_to_idx", vocab)
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx.get("<PAD>", 0)

    # Initialiser modèle
    max_len = args.MAX_LENGTH + 2
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    max_len, max_len,
                    args.d_model, args.num_heads, args.dff, 0.1).to(device)

    # Initialiser optimiseur et loss
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=0.001, betas=(0.9, 0.98),
                                 eps=1e-8, weight_decay=5e-4)

    # Initialiser les poids
    initNetParams(deepsc)

    # Créer répertoire checkpoints
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Phase 1 : LR = 0.001
    for epoch in range(100):
        for g in optimizer.param_groups:
            g['lr'] = 0.001
        print(f"🚀 Phase 1 | Epoch {epoch+1}/200 | LR = 0.001")
        train(epoch, args, deepsc)
        checkpoint_file = os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(deepsc.state_dict(), checkpoint_file)
        print(f"✅ Checkpoint sauvegardé : {checkpoint_file}")

    # Phase 2 : LR = 0.002
    for epoch in range(100, 200):
        for g in optimizer.param_groups:
            g['lr'] = 0.002
        print(f"🚀 Phase 2 | Epoch {epoch+1}/200 | LR = 0.002")
        train(epoch, args, deepsc)
        checkpoint_file = os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(deepsc.state_dict(), checkpoint_file)
        print(f"✅ Checkpoint sauvegardé : {checkpoint_file}")
