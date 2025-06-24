import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi_step
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==================== ARGUMENTS ====================
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc_complex_AWGN', type=str)
parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=100, type=int)

# ==================== DEVICE ====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==================== SEED ====================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ==================== VALIDATION ====================
def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_loader = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                             pin_memory=True, collate_fn=collate_data)
    net.eval()
    total = 0
    with torch.no_grad():
        for sents in tqdm(test_loader, desc=f"Epoch {epoch+1} - VAL"):
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx, criterion, args.channel)
            total += loss
    return total / len(test_loader)

# ==================== ENTRAINEMENT ====================
def train(epoch, args, net, mi_net=None):
    train_eur = EurDataset('train')
    train_loader = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                              pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_loader)

    noise_std = SNR_to_noise(12)  # Bruit pour un SNR fixe à 12dB

    total_loss, total_ce, total_mi, count = 0, 0, 0, 0

    for sents in pbar:
        sents = sents.to(device)

        # Entraînement MI
        mi = 0
        if mi_net is not None:
            mi = train_mi_step(net, mi_net, sents, noise_std, pad_idx, mi_opt, args.channel)

        # Entraînement DeepSC
        if mi_net is not None:
            loss_total, loss_ce, loss_mi = train_step(
                net, sents, sents, noise_std, pad_idx,
                optimizer, criterion, args.channel, mi_net
            )
        else:
            loss_ce = train_step(net, sents, sents, noise_std, pad_idx,
                                 optimizer, criterion, args.channel)
            loss_total = loss_ce
            loss_mi = 0

        total_loss += loss_total
        total_ce += loss_ce
        total_mi += loss_mi
        count += 1

        pbar.set_description(
            f"Epoch {epoch+1}; Loss: {loss_total:.4f}; CE: {loss_ce:.4f}; MI: {loss_mi:.4f}; Noise: {noise_std:.4f}"
        )

    return total_loss / count, total_ce / count, total_mi / count

# ==================== MAIN ====================
if __name__ == '__main__':
    args = parser.parse_args()

    vocab = json.load(open(args.vocab_file, 'r'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx['<PAD>']
    start_idx = token_to_idx['<START>']
    end_idx = token_to_idx['<END>']

    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)
    mi_net = Mine().to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)

    initNetParams(deepsc)

    best_val_loss = float('inf')
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    loss_curve, ce_curve, mi_curve = [], [], []

    for epoch in range(args.epochs):
        train_loss, train_ce, train_mi = train(epoch, args, deepsc, mi_net)
        val_loss = validate(epoch, args, deepsc)

        loss_curve.append(train_loss)
        ce_curve.append(train_ce)
        mi_curve.append(train_mi)

        # --- Sauvegarder tous les checkpoints ---
        torch.save(deepsc.state_dict(), os.path.join(
            args.checkpoint_path,
            f'checkpoint_epoch{epoch+1:03d}_loss{train_loss:.4f}_ce{train_ce:.4f}_mi{train_mi:.4f}.pth' ))

        # --- Sauvegarder le meilleur modèle ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(deepsc.state_dict(), os.path.join(
                args.checkpoint_path, 'best_model.pth'))

    # === Save and plot curves ===
    np.save(os.path.join(args.checkpoint_path, 'loss_curve.npy'), np.array(loss_curve))
    np.save(os.path.join(args.checkpoint_path, 'ce_curve.npy'), np.array(ce_curve))
    np.save(os.path.join(args.checkpoint_path, 'mi_curve.npy'), np.array(mi_curve))

    plt.figure()
    plt.plot(loss_curve, label='Total Loss (CE - λ * MI)')
    plt.plot(ce_curve, label='Cross-Entropy Loss (CE)')
    plt.plot(mi_curve, label='Mutual Information (MI)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_path, 'training_curves.png'))
    plt.show()
