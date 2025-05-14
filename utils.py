import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 🔁 Masque de décodage : empêche de voir les tokens futurs
def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask)


# ✅ Créer masques pour l'attention
def create_masks(src, trg, padding_idx):
    src_mask = (src != padding_idx).unsqueeze(1).type(torch.float32)
    trg_mask = (trg != padding_idx).unsqueeze(1).type(torch.float32)

    look_ahead_mask = subsequent_mask(trg.size(1)).to(trg.device).type(torch.float32)
    combined_mask = torch.max(trg_mask, look_ahead_mask)

    return src_mask.to(device), combined_mask.to(device)


# ✅ Fonction de loss avec padding ignoré
def loss_function(logits, targets, padding_idx, criterion):
    loss = criterion(logits, targets)  # shape [batch * seq_len]
    mask = (targets != padding_idx).type_as(loss)  # 1 pour les vrais tokens, 0 pour <PAD>
    loss = loss * mask  # on ignore les PAD
    return loss.sum() / mask.sum()  # moyenne sur les vrais tokens



# ✅ Initialisation des poids
def initNetParams(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ✅ Étape d'entraînement (1 batch)
def train_step(model, src, trg, pad, optimizer, criterion):
    model.train()

    trg_input = trg[:, :-1]
    trg_real = trg[:, 1:]

    optimizer.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_input, pad)

    outputs = model(src, trg_input, src_mask=src_mask, look_ahead_mask=look_ahead_mask)

    ntokens = outputs.size(-1)
    loss = loss_function(outputs.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    loss.backward()
    optimizer.step()

    return loss


# ✅ Étape de validation (1 batch)
def val_step(model, src, trg, pad, criterion):
    model.eval()

    trg_input = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_input, pad)

    with torch.no_grad():
        outputs = model(src, trg_input, src_mask=src_mask, look_ahead_mask=look_ahead_mask)

    ntokens = outputs.size(-1)
    loss = loss_function(outputs.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    return loss
