import torch
import json
from models.transceiver import DeepSC

# -------- PARAMÈTRES DU MODÈLE --------
d_model = 128
num_layers = 4
num_heads = 8
dff = 512
max_len = 32  # 30 + 2
checkpoint_path = 'checkpoints/deepsc-NoChannel/checkpoint_epoch_42.pth'

# -------- CHARGER LE VOCABULAIRE --------
with open('data/vocab.json', 'r', encoding='utf-8') as f:
    token_to_idx = json.load(f)

idx_to_token = {v: k for k, v in token_to_idx.items()}

pad_idx = token_to_idx.get("<PAD>", 0)
sos_idx = token_to_idx.get("<START>", 1)
eos_idx = token_to_idx.get("<END>", 2)
unk_idx = token_to_idx.get("<UNK>", 3)

vocab_size = len(token_to_idx)

# -------- FONCTIONS --------
def encode_sentence(sentence, token_to_idx, max_len):
    tokens = ["<START>"] + sentence.lower().split() + ["<END>"]
    token_ids = [token_to_idx.get(token, unk_idx) for token in tokens]
    token_ids += [pad_idx] * (max_len - len(token_ids))
    return torch.tensor(token_ids).unsqueeze(0)  # [1, seq_len]

def greedy_decode(model, input_sentence, max_len, sos_idx, eos_idx):
    model.eval()
    input_tensor = encode_sentence(input_sentence, token_to_idx, max_len).to(device)

    output_ids = [sos_idx]
    for _ in range(max_len):
        trg_tensor = torch.tensor(output_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor, trg_tensor)
        next_token = output[0, -1].argmax(-1).item()

        if next_token == eos_idx:
            break
        output_ids.append(next_token)

    return " ".join([idx_to_token.get(tok, "<UNK>") for tok in output_ids[1:]])

# -------- CHARGER LE MODÈLE --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepSC(num_layers, vocab_size, vocab_size,
               max_len, max_len, d_model, num_heads, dff).to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# -------- TESTER UNE PHRASE --------
test_sentence = "bonjour comment ça va"
output_sentence = greedy_decode(model, test_sentence, max_len, sos_idx, eos_idx)

print(" Phrase d'entrée :", test_sentence)
print(" Phrase générée :", output_sentence)
