import os
import argparse
import json
import torch
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SNR_to_noise, val_step

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-NoChannel', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help='Please choose AWGN, Rayleigh')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Fonction de validation
def validate(epoch, args, net, mi_net=None):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    
    # Variables pour enregistrer les pertes
    cross_entropy_losses = []
    mi_losses = []
    total_losses = []

    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)

            # Calcul de la perte de cross-entropy
            ce_loss = val_step(net, sents, sents, pad_idx, criterion)

            # Si un réseau MI est donné, calculer la perte MI
            if mi_net is not None:
                z = net(sents)
                mi_loss = mi_net(z)
                total_loss = ce_loss + lambda_mi * mi_loss  # Combinaison des pertes
                pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; CE Loss: {ce_loss:.5f}; MI Loss: {mi_loss:.5f}; Total Loss: {total_loss:.5f}')
            else:
                total_loss = ce_loss
                pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; CE Loss: {ce_loss:.5f}; Total Loss: {total_loss:.5f}')
            
            # Enregistrer les pertes
            cross_entropy_losses.append(ce_loss.item())
            mi_losses.append(mi_loss.item() if mi_net is not None else 0)
            total_losses.append(total_loss.item())

    # Retourner les pertes
    return cross_entropy_losses, mi_losses, total_losses

# Point d'entrée principal pour tester
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

    # Charger le modèle sauvegardé
    checkpoint_path = os.path.join(args.checkpoint_path, f'checkpoint_epoch_{args.epochs}.pth')
    deepsc.load_state_dict(torch.load(checkpoint_path))
    
    validate(args.epochs, args, deepsc, mi_net)
