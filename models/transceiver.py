import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ---------- Multi-Head Attention ----------
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        B = query.size(0)

        query = self.wq(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        key   = self.wk(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.wv(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(p_attn, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.dropout(self.dense(attn_output))

# ---------- Position-wise Feedforward ----------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(F.relu(self.linear1(x))))

# ---------- Encoder Layer ----------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.norm1(x + self.mha(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return x

# ---------- Decoder Layer ----------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout)
        self.enc_dec_mha = MultiHeadedAttention(num_heads, d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, look_ahead_mask, padding_mask):
        x = self.norm1(x + self.self_mha(x, x, x, look_ahead_mask))
        x = self.norm2(x + self.enc_dec_mha(x, memory, memory, padding_mask))
        x = self.norm3(x + self.ffn(x))
        return x

# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self, num_layers, vocab_size, max_len, d_model, num_heads, dff, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# ---------- Decoder ----------
class Decoder(nn.Module):
    def __init__(self, num_layers, vocab_size, max_len, d_model, num_heads, dff, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.d_model = d_model

    def forward(self, x, memory, look_ahead_mask, padding_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, look_ahead_mask, padding_mask)
        return x

# ---------- DeepSC (sans canal) ----------
class DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len, trg_max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(DeepSC, self).__init__()
        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len, d_model, num_heads, dff, dropout)
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len, d_model, num_heads, dff, dropout)
        self.dense = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg_input, src_mask=None, trg_mask=None, look_ahead_mask=None):
        enc_output = self.encoder(src, src_mask)                    # [B, L, d_model]
        dec_output = self.decoder(trg_input, enc_output, look_ahead_mask, trg_mask)  # [B, L, d_model]
        logits = self.dense(dec_output)                             # [B, L, vocab_size]
        return logits
