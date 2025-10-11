import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NMTRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, 
                 n_layers, dropout=None, input_feeding=False,
                 attention_mode='no', attention_win=10, max_length=50,
                 device='cpu', padding_idx=0):
        super(NMTRNN, self).__init__()
        self.attention_mode = attention_mode
        self.encoder = LSTMEncoder(vocab_size=input_size, embed_dim=embedding_dim, hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout, padding_idx=padding_idx, device=device)
        self.decoder = LSTMDecoder(output_dim=output_size, embed_dim=embedding_dim, hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout, attention_mode=attention_mode, attention_win=attention_win, max_length=max_length, padding_idx=padding_idx, device=device)
        self.init_weights()
        self.device = device

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.1, 0.1)

    def forward(self, src, trg_input, src_lengths=None, mode='train'):
        trg_length = trg_input.shape[0]
        batch_size = trg_input.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        encoder_output, (hidden, cell) = self.encoder(src, src_lengths)

        if self.attention_mode == 'no': encoder_output = None

        decoder_input = trg_input[0,:]
        for t in range(trg_length):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_output)
            outputs[t] = decoder_output.squeeze(0) # shape (1, batch_size, trg_vocab_size)

            if t == trg_length-1: break
            if mode=='train' or mode=='test':
                decoder_input = trg_input[t+1,:]
            elif mode=='generate':
                decoder_input = decoder_output.argmax(2).squeeze(0)
        
        return outputs

    def predict(self, src, trg_input):
        self.eval()
        with torch.no_grad():
            output = self.forward(src, trg_input, mode='generate')
        return output

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, padding_idx=0, device='cpu'):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)

    def forward(self, text, lengths=None):
        embedded = self.embedding(text)
        if lengths is not None:
            packed_embed = pack_padded_sequence(embedded, lengths, batch_first=False, enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_embed)
            output, _ = pad_packed_sequence(packed_output, batch_first=False)
        else:
            output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.clamp(hidden, min=-50, max=50)
        cell = torch.clamp(cell, min=-50, max=50)
        return output, (hidden, cell)
    
class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout, attention_mode, attention_win, max_length=50, padding_idx=0, device='cpu'):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.attn = attention_mode
        self.attn_win = attention_win
        self.device=device

        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=padding_idx)
        if attention_mode == 'no':
            self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
        elif attention_mode == 'global':
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
            self.attention = Attention(mode='global', hidden_dim=hidden_dim, src_length=max_length, device=device)
        elif attention_mode == 'local':
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
            self.attention = Attention(mode='local', hidden_dim=hidden_dim, src_length=attention_win, device=device)

        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, input, hidden, cell, encoder_output=None):

        if input.dim() == 1:
            input = input.unsqueeze(0)

        embedded = self.embedding(input)

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        if self.attn == 'global':
            output = self.attention(encoder_output, hidden[-1])
        elif self.attn == 'local':
            output = self.attention(encoder_output, hidden[-1])

        hidden = torch.clamp(hidden, min=-50, max=50)
        cell = torch.clamp(cell, min=-50, max=50)

        output = self.fc_out(output)
        return output, hidden, cell
    

class Attention(nn.Module):
    def __init__(self, mode='global', hidden_dim=1000, src_length=50, device='cpu'):
        super(Attention, self).__init__()
        self.mode = mode
        self.device=device
        
        if mode == 'global':
            self.align = self.align_location
            self.W_a = nn.Linear(hidden_dim, src_length, bias=False)
            self.W_c = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        elif mode =='local':
            self.align = self.align_general
            self.attention_win = src_length
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v_p = nn.Linear(hidden_dim, 1, bias=False)
            self.W_c = nn.Linear(hidden_dim*2, hidden_dim, bias=False)

    def forward(self, encoder_hidden, target_hidden):
        src_length, batch_size, hidden_dim = encoder_hidden.shape

        # calculate alignment score
        align, p_t = self.align(target_hidden, encoder_hidden) # (src_length, batch_size)
        align = align.T

        # calculate context vector by weighted averaging
        # Broadcast?
        if self.mode == 'global':
            context = torch.mul(encoder_hidden, align[:src_length].view(src_length, batch_size, 1)) # (src_length, batch_size, hidden_dim) * (src_length, batch_size) -> (src_length, batch_size, hidden_dim)
        if self.mode == 'local':
            context = torch.mul(encoder_hidden, align[:src_length].view(src_length, batch_size, 1))
        context = torch.sum(context, dim=0) # -> (batch_size, hidden_dim)

        # concatenate context and hidden_state
        context = torch.cat((context, target_hidden), dim=1) # -> (batch_size, 2*hidden_dim)
        context = self.W_c(context)
        attention_hidden = F.tanh(context)

        return attention_hidden

    def align_general(self, trg_hidden, src_hidden): # local attention
        # trg_hidden : (batch_size, hidden_dim)
        # src_hidden : (src_length, batch_size, hidden_dim) TODO: need to check shapes

        S, batch_size, hidden_dim = src_hidden.shape

        position_t = self.W_p(trg_hidden) # -> (batch_size, hidden_dim)
        position_t = F.tanh(position_t)
        position_t = S * F.sigmoid(self.v_p(position_t)) # -> (batch_size, 1)

        score = self.W_a(src_hidden) # (src_length, batch_size, hidden_dim)
        
        score = torch.mul(trg_hidden.expand(S,-1,-1), score) # (batch_size, hidden_dim) * (src_length, batch_size, hidden_dim) -> (src_length, batch_size)
        score = torch.sum(score, dim=2)
        align = F.softmax(score, dim=1).T # (batch_size, src_length)

        # build mask
        src_indices = torch.arange(S, device=self.device).unsqueeze(0).repeat(batch_size, 1) # (batch_size, S)
        lower_bound = position_t - self.attention_win # (batch_size, 1)
        upper_bound = position_t + self.attention_win
        mask = (src_indices >= lower_bound) & (src_indices < upper_bound) # (batch_size, S)

        # calculate gaussian values for each position within window
        gaussian = torch.exp(-((src_indices-position_t)**2 / 2*(self.attention_win/2)**2)) # (batch_size, S)

        align = torch.mul(align, gaussian) # (batch_size, src_length)
        align = align*mask
        return align, position_t

    def align_location(self, trg_hidden, src_hidden=None): # global attention
        # trg_hidden : (batch_size, hidden_dim)

        score = self.W_a(trg_hidden) # (batch_size, src_length)
        align = F.softmax(score, dim=1)
        return align, None
    