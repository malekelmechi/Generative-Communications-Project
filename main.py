import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
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
parser.add_argument('--epochs', default=80, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, pad_idx,
                            criterion)
            total += loss
            pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')
    return total / len(test_iterator)

def train(epoch, args, net, mi_net=None):
    train_eur = EurDataset('train')
    print(f" Loaded train dataset with {len(train_eur)} samples")  

    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    print(" Created DataLoader") 

    pbar = tqdm(train_iterator)


    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(f'Epoch: {epoch + 1}; Type: Train; Loss: {loss:.5f}; MI: {mi:.5f}')
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)
            pbar.set_description(f'Epoch: {epoch + 1}; Type: Train; Loss: {loss:.5f}')

if __name__ == '__main__':
    args = parser.parse_args()


    vocab_path = os.path.join(os.path.dirname(__file__), args.vocab_file)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    token_to_idx = vocab.get("token_to_idx", vocab)

    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx.get("<PAD>", 0)
    start_idx = token_to_idx.get("<START>", 1)
    end_idx = token_to_idx.get("<END>", 2)

   
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)

    initNetParams(deepsc)

    record_acc = float('inf')
    for epoch in range(args.epochs):
        train(epoch, args, deepsc)
        avg_acc = validate(epoch, args, deepsc)

        if avg_acc < record_acc:
            os.makedirs(args.checkpoint_path, exist_ok=True)
            checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_{str(epoch + 1).zfill(2)}.pth')
            torch.save(deepsc.state_dict(), checkpoint_file)
            record_acc = avg_acc
