import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-AWGN', type=str)
parser.add_argument('--channel', default='AWGN', type=str)  # peut changer AWGN/Rayleigh
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Appareil utilisé : {device}")


def performance(args, SNR_list, model, token_to_idx, pad_idx, start_idx, end_idx):
    

    bleu1 = BleuScore(1, 0, 0, 0)
    bleu2 = BleuScore(0.5, 0.5, 0, 0)
    bleu3 = BleuScore(0.33, 0.33, 0.33,0)
    bleu4 = BleuScore(0.25, 0.25, 0.25, 0.25)


    test_set = EurDataset('test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_data,num_workers=0)

    StoT = SeqtoText(token_to_idx, end_idx)

    bleu1_results, bleu2_results, bleu3_results , bleu4_results= [], [], [],[]

    model.eval()
    with torch.no_grad():
        for snr in tqdm(SNR_list, desc="SNR Loop"):
            noise_std = SNR_to_noise(snr)

            pred_sentences = []
            target_sentences = []

            for batch in test_loader:
                
                batch = batch.to(device)
                target = batch

                output = greedy_decode(model, batch, noise_std, args.MAX_LENGTH, pad_idx, start_idx, args.channel)

                pred_strings = list(map(StoT.sequence_to_text, output.cpu().numpy().tolist()))
                target_strings = list(map(StoT.sequence_to_text, target.cpu().numpy().tolist()))

                pred_sentences.extend(pred_strings)
                target_sentences.extend(target_strings)

            bleu1_score = np.mean(bleu1.compute_blue_score(pred_sentences, target_sentences))
            bleu2_score = np.mean(bleu2.compute_blue_score(pred_sentences, target_sentences))
            bleu3_score = np.mean(bleu3.compute_blue_score(pred_sentences, target_sentences))
            bleu4_score = np.mean(bleu4.compute_blue_score(pred_sentences, target_sentences))

            bleu1_results.append(bleu1_score)
            bleu2_results.append(bleu2_score)
            bleu3_results.append(bleu3_score)
            bleu4_results.append(bleu4_score)

            print(f"\n[SNR {snr} dB] BLEU-1: {bleu1_score:.4f}, BLEU-2: {bleu2_score:.4f}, BLEU-3: {bleu3_score:.4f}, , BLEU-4: {bleu4_score:.4f}")

    return bleu1_results, bleu2_results, bleu3_results, bleu4_results

if __name__ == '__main__':
    args = parser.parse_args()

    SNR_values = [0, 3, 6, 9, 12, 15, 18]

    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    model = DeepSC(
        args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
        args.d_model, args.num_heads, args.dff, 0.1
    ).to(device)

    # Load latest checkpoint
    checkpoints = [
        (os.path.join(args.checkpoint_path, fn), int(fn.split('_')[-1].split('.')[0]))
        for fn in os.listdir(args.checkpoint_path) if fn.endswith('.pth')
    ]

    model_path, _ = sorted(checkpoints, key=lambda x: x[1])[-1]
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)


    print(f" Modèle chargé depuis {model_path}")

    bleu1, bleu2, bleu3, bleu4 = performance(args, SNR_values, model, token_to_idx, pad_idx, start_idx, end_idx)

    print("\n Résultats finaux BLEU pour chaque SNR :")
    for snr, b1, b2, b3, b4 in zip(SNR_values, bleu1, bleu2, bleu3, bleu4):
        print(f"SNR {snr} dB - BLEU-1: {b1:.4f}, BLEU-2: {b2:.4f}, BLEU-3: {b3:.4f}, BLEU-4: {b4:.4f}")

    # Tracer les courbes BLEU-1, BLEU-2, BLEU-3
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_values, bleu1, marker='o', label="BLEU-1 (1-gram)")
    plt.plot(SNR_values, bleu2, marker='s', label="BLEU-2 (2-gram)")
    plt.plot(SNR_values, bleu3, marker='^', label="BLEU-3 (3-gram)")
    plt.plot(SNR_values, bleu4,  label="BLEU-4 (4-gram)")
    plt.title(f'Courbe BLEU vs SNR ({args.channel} Channel)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Score BLEU')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bleu_scores_vs_snr_{args.channel}.png")
    plt.show()