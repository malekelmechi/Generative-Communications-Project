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
parser.add_argument('--epochs', default=84, type=int)
print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nom du GPU :", torch.cuda.get_device_name(0))

# \uD83D\uDD01 Fine-tuning et reprise
parser.add_argument('--resume', default=None, type=str, help='Chemin du checkpoint à charger pour fine-tuning')
parser.add_argument('--fine-tune-lr', default=None, type=float, help='Learning rate pour le fine-tuning')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

    print(f"Loss (avg) saved for epoch {epoch}: {avg_loss:.4f}")

def train(epoch, args, net):
    net.train()
    train_eur = EurDataset('train')
    print(f"Loaded train dataset with {len(train_eur)} samples")

    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)

    pbar = tqdm(train_iterator)
    cross_entropy_losses = []

    for sents in pbar:
        sents = sents.to(device)
        input_ids = sents[:, :-1]
        target_ids = sents[:, 1:]

        ce_loss = train_step(net, input_ids, target_ids, pad_idx, optimizer, criterion)

        pbar.set_description(f'Epoch: {epoch + 1}; Train CE Loss: {ce_loss:.5f}')
        cross_entropy_losses.append(ce_loss.item())

    save_losses(cross_entropy_losses, epoch)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    token_to_idx = vocab.get("token_to_idx", vocab)
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx.get("<PAD>", 0)

    max_len = args.MAX_LENGTH + 2
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    max_len, max_len,
                    args.d_model, args.num_heads, args.dff, 0.1).to(device)

    # ✅ Charger un checkpoint si demandé
    if args.resume:
        print(f" Reprise depuis : {args.resume}")
        deepsc.load_state_dict(torch.load(args.resume, map_location=device))

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98),
                                 eps=1e-8, weight_decay=5e-4)

    # ✅ Changer le learning rate pour fine-tuning
    if args.fine_tune_lr:
        print(f"7 Changement du learning rate : {args.fine_tune_lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.fine_tune_lr

    initNetParams(deepsc)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    for epoch in range(args.epochs):
        train(epoch, args, deepsc)

        checkpoint_file = os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(deepsc.state_dict(), checkpoint_file)
        print(f"✅ Checkpoint enregistré : {checkpoint_file}")