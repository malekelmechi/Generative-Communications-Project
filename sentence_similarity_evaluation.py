import os
# ⚠️ Doit être avant les imports Transformers pour forcer PyTorch seulement
os.environ["USE_TF"] = "0"

import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-AWGN', type=str)
parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sentence_similarity(args, SNR_list, model, token_to_idx, pad_idx, start_idx, end_idx):
    StoT = SeqtoText(token_to_idx, end_idx)
    test_set = EurDataset('test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_data, num_workers=0)

    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    sim_scores_by_snr = []

    model.eval()
    with torch.no_grad():
        for snr in tqdm(SNR_list, desc="SNR Loop"):
            noise_std = SNR_to_noise(snr)
            pred_sentences, target_sentences = [], []

            for batch in test_loader:
                batch = batch.to(device)
                output = greedy_decode(model, batch, noise_std, args.MAX_LENGTH, pad_idx, start_idx, args.channel)

                pred_strings = list(map(StoT.sequence_to_text, output.cpu().numpy().tolist()))
                target_strings = list(map(StoT.sequence_to_text, batch.cpu().numpy().tolist()))

                pred_sentences.extend(pred_strings)
                target_sentences.extend(target_strings)

            # Embedding et similarité
            embeddings_pred = bert_model.encode(pred_sentences, convert_to_tensor=True)
            embeddings_target = bert_model.encode(target_sentences, convert_to_tensor=True)

            cos_sim = util.cos_sim(embeddings_pred, embeddings_target)
            sim_scores = cos_sim.diag()
            mean_score = sim_scores.mean().item()
            sim_scores_by_snr.append(mean_score)

            print(f"\n🔊 [SNR {snr} dB] Similarité moyenne : {mean_score:.4f}")

            # ✅ Affichage de 3 exemples
            print("\n📌 Exemples (Original vs Transmis) :")
            for i in range(3):
                print(f"\nExemple {i+1}:")
                print("  ➤ Original :", target_sentences[i])
                print("  ➤ Transmis :", pred_sentences[i])
                print("  ➤ Similarité cosine :", f"{cos_sim[i, i].item():.4f}")

    return sim_scores_by_snr


if __name__ == "__main__":
    args = parser.parse_args()
    SNR_values = [0, 3, 6, 9, 12, 15, 18]

    # Vocabulaire
    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    num_vocab = len(token_to_idx)

    # Chargement modèle
    model = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                   args.d_model, args.num_heads, args.dff, 0.1).to(device)

    checkpoints = [
        (os.path.join(args.checkpoint_path, fn), int(fn.split('_')[-1].split('.')[0]))
        for fn in os.listdir(args.checkpoint_path) if fn.endswith('.pth')
    ]

    model_path, _ = sorted(checkpoints, key=lambda x: x[1])[-1]
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Évaluation
    sim_scores = sentence_similarity(args, SNR_values, model, token_to_idx, pad_idx, start_idx, end_idx)

    # Résultats
    print("\n📊 Résumé final des similarités de phrases :")
    for snr, score in zip(SNR_values, sim_scores):
        print(f"SNR {snr} dB - Sentence Similarity: {score:.4f}")

    # Courbe
    plt.figure()
    plt.plot(SNR_values, sim_scores, marker='o', label='Sentence Similarity')
    plt.title(f'Sentence Similarity vs SNR ({args.channel} channel)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"sentence_similarity_vs_snr_{args.channel}.png")
    plt.show()
