import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from models.mutual_info import sample_batch, mutual_information

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words) 




class Channels():
    def AWGN(self, Tx_sig_complex, noise_variance):
        """
        Tx_sig_complex : tensor complexe [batch, ..., 16] (ou autre)
        noise_variance : variance du bruit (float)
        """
        noise_real = torch.normal(0, math.sqrt(noise_variance / 2), size=Tx_sig_complex.size()).to(device)
        noise_imag = torch.normal(0, math.sqrt(noise_variance / 2), size=Tx_sig_complex.size()).to(device)
        noise = torch.complex(noise_real, noise_imag)
        Rx_sig = Tx_sig_complex + noise
        return Rx_sig

    def Rayleigh(self, Tx_sig_complex, noise_variance):
        shape = Tx_sig_complex.shape
        # Canaux complexes Rayleigh : H ~ CN(0,1)
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.complex(H_real, H_imag)  # scalaire complexe

        # Appliquer le canal complexe à chaque vecteur du batch :
        # On suppose que le dernier dim est la dimension du signal complexe
        # Tx_sig_complex : [batch, ..., dim]
        Rx_sig = Tx_sig_complex * H  # multiplication élément par élément par un scalaire complexe

        # Ajouter bruit AWGN
        Rx_sig = self.AWGN(Rx_sig, noise_variance)

        # Estimation canal (inversion)
        Rx_sig = Rx_sig / H

        return Rx_sig

    def Rician(self, Tx_sig_complex, noise_variance, K=1):
        shape = Tx_sig_complex.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.complex(H_real, H_imag)

        Rx_sig = Tx_sig_complex * H
        Rx_sig = self.AWGN(Rx_sig, noise_variance)
        Rx_sig = Rx_sig / H

        return Rx_sig

def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
         
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

    
def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    
    return src_mask.to(device), combined_mask.to(device)

def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

def train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()
    
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    #Tx_sig = quantize_complex_signal(Tx_sig, nb_bit_frac=6, nb_bit_int=6)
    Tx_sig_real = Tx_sig[:, :, :16]
    Tx_sig_imag = Tx_sig[:, :, 16:]
    Tx_sig_complex = torch.complex(Tx_sig_real, Tx_sig_imag)

    if channel == 'AWGN':
        Rx_sig_complex = channels.AWGN(Tx_sig_complex, n_var)
    elif channel == 'Rayleigh':
        Rx_sig_complex = channels.Rayleigh(Tx_sig_complex, n_var)
    elif channel == 'Rician':
        Rx_sig_complex = channels.Rician(Tx_sig_complex, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    
    Rx_sig_real = Rx_sig_complex.real
    Rx_sig_imag = Rx_sig_complex.imag
    Rx_sig = torch.cat([Rx_sig_real, Rx_sig_imag], dim=-1)
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    
    ntokens = pred.size(-1)
    
    # Cross-entropy loss
    loss_ce = loss_function(pred.contiguous().view(-1, ntokens), 
                            trg_real.contiguous().view(-1), 
                            pad, criterion)

    # Mutual Information loss (optional)
    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig_complex, Rx_sig_complex)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mi = mi_lb
        lambda_mi =0.0009
        loss_total = loss_ce - lambda_mi * loss_mi
    else:
        loss_mi = torch.tensor(0.0).to(device)
        loss_total = loss_ce

    loss_total.backward()
    opt.step()

    return loss_total.item(), loss_ce.item(), loss_mi.item()


def train_mi_step(model, mi_net, src, n_var, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    
    Tx_sig_real = Tx_sig[:, :, :16]
    Tx_sig_imag = Tx_sig[:, :, 16:]
    Tx_sig_complex = torch.complex(Tx_sig_real, Tx_sig_imag)

    if channel == 'AWGN':
        Rx_sig_complex = channels.AWGN(Tx_sig_complex, n_var)
    elif channel == 'Rayleigh':
        Rx_sig_complex = channels.Rayleigh(Tx_sig_complex, n_var)
    elif channel == 'Rician':
        Rx_sig_complex = channels.Rician(Tx_sig_complex, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    Rx_sig_real = Rx_sig_complex.real
    Rx_sig_imag = Rx_sig_complex.imag
    Rx_sig = torch.cat([Rx_sig_real, Rx_sig_imag], dim=-1)
   
    joint, marginal = sample_batch(Tx_sig_complex, Rx_sig_complex)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()

    return loss_mine.item()

def val_step(model, src, trg, n_var, pad, criterion, channel):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    #Tx_sig = quantize_complex_signal(Tx_sig, nb_bit_frac=6, nb_bit_int=6)

    Tx_sig_real = Tx_sig[:, :, :16]
    Tx_sig_imag = Tx_sig[:, :, 16:]
    Tx_sig_complex = torch.complex(Tx_sig_real, Tx_sig_imag)

    if channel == 'AWGN':
        Rx_sig_complex = channels.AWGN(Tx_sig_complex, n_var)
    elif channel == 'Rayleigh':
        Rx_sig_complex = channels.Rayleigh(Tx_sig_complex, n_var)
    elif channel == 'Rician':
        Rx_sig_complex = channels.Rician(Tx_sig_complex, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    Rx_sig_real = Rx_sig_complex.real
    Rx_sig_imag = Rx_sig_complex.imag
    Rx_sig = torch.cat([Rx_sig_real, Rx_sig_imag], dim=-1)
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)

    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    # loss = loss_function(pred, trg_real, pad)
    
    return loss.item()
    
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    """ 
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)


    Tx_sig_real = Tx_sig[:, :, :16]
    Tx_sig_imag = Tx_sig[:, :, 16:]
    Tx_sig_complex = torch.complex(Tx_sig_real, Tx_sig_imag)

    if channel == 'AWGN':
        Rx_sig_complex = channels.AWGN(Tx_sig_complex, n_var)
    elif channel == 'Rayleigh':
        Rx_sig_complex = channels.Rayleigh(Tx_sig_complex, n_var)
    elif channel == 'Rician':
        Rx_sig_complex = channels.Rician(Tx_sig_complex, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    Rx_sig_real = Rx_sig_complex.real
    Rx_sig_imag = Rx_sig_complex.imag
    Rx_sig = torch.cat([Rx_sig_real, Rx_sig_imag], dim=-1)

    #channel_enc_output = model.blind_csi(channel_enc_output)
          
    memory = model.channel_decoder(Rx_sig)
    
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
#        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        
        # predict the word
        prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
        #prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim = -1)
        #next_word = next_word.unsqueeze(1)
        
        #next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs



import torch

def quantize_complex_fixed_point(x, nb_bit_frac=6, nb_bit_int=6, eps=1e-8):
    real = x.real
    imag = x.imag

    alpha_real = 1.0 / torch.sqrt(torch.mean(real**2) + eps)
    alpha_imag = 1.0 / torch.sqrt(torch.mean(imag**2) + eps)

    real_norm = real * alpha_real
    imag_norm = imag * alpha_imag

    base = 2 ** (-nb_bit_frac)
    val_max = 2 ** (nb_bit_int - 1) - base
    val_min = -2 ** (nb_bit_int - 1)

    real_q = torch.clamp(torch.round(real_norm / base) * base, val_min, val_max)
    imag_q = torch.clamp(torch.round(imag_norm / base) * base, val_min, val_max)

    real_out = real_q / alpha_real
    imag_out = imag_q / alpha_imag

    return torch.complex(real_out, imag_out)
