# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Toi
@File: performance_ss.py
@Role: Évaluation de la Similarité de phrases avec DeepSC et BERT
"""

import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from utils import SNR_to_noise, greedy_decode, SeqtoText
from similarity import Similarity  # fichier séparé

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-AWGN_lambda0.001', type=str)
parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--bert-config-path', default='uncased_L-12_H-768_A-12/bert_config.json', type=str)
parser.add_argument('--bert-checkpoint-path', default='uncased_L-12_H-768_A-12/bert_model.ckpt', type=str)
parser.add_argument('--bert-dict-path', default='uncased_L-12_H-768_A-12/vocab.txt', type=str)

def compute_sentence_similarity(args, SNR, model):
    similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)

    test_set = EurDataset('test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0,
                             pin_memory=True, collate_fn=collate_data)

    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    end_idx = token_to_idx["<END>"]
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]

    to_text = SeqtoText(token_to_idx, end_idx)
    model.eval()

    all_scores = []

    with torch.no_grad():
        for snr in tqdm(SNR, desc="SNR Loop"):
            predictions = []
            targets = []
            noise_std = SNR_to_noise(snr)

            for batch in test_loader:
                batch = batch.to(device)
                decoded = greedy_decode(model, batch, noise_std, args.MAX_LENGTH,
                                        pad_idx, start_idx, args.channel)

                pred_texts = list(map(to_text.sequence_to_text, decoded.cpu().numpy().tolist()))
                target_texts = list(map(to_text.sequence_to_text, batch.cpu().numpy().tolist()))

                predictions.extend(pred_texts)
                targets.extend(target_texts)

            scores = similarity.compute_similarity(predictions, targets)
            mean_score = np.mean(scores)
            all_scores.append(mean_score)
            print(f"SNR {snr} dB → Mean Sentence Similarity: {mean_score:.4f}")

    return all_scores


if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]

    vocab = json.load(open(args.vocab_file, 'rb'))
    num_vocab = len(vocab['token_to_idx'])

    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                    d_model=128, num_heads=8, dff=512, dropout_rate=0.1).to(device)

    # Charger le dernier checkpoint
    model_files = [f for f in os.listdir(args.checkpoint_path) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("Aucun checkpoint trouvé dans le dossier.")

    latest_model = sorted(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    checkpoint = torch.load(os.path.join(args.checkpoint_path, latest_model), map_location=device)
    deepsc.load_state_dict(checkpoint)
    print("✅ Modèle chargé :", latest_model)

    sim_scores = compute_sentence_similarity(args, SNR, deepsc)
    print("📊 Similarité moyenne par SNR :", sim_scores)
