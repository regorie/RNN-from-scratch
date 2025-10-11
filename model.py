import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NTRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, 
                 n_layers, dropout=None, attention_mode='no', attention_win=10, max_length=50,
                 device='cpu', padding_idx=0):
        super(NTRNN, self).__init__()
        self.attention_mode = attention_mode
        self.encoder = LSTMEncoder(vocab_size=input_size, embed_dim=embedding_dim, hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout, padding_idx=padding_idx)
        self.decoder = LSTMDecoder(output_dim=output_size, embed_dim=embedding_dim, hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout, attention_mode=attention_mode, attention_win=attention_win, max_length=max_length, padding_idx=padding_idx)
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
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, padding_idx=0):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

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
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout, attention_mode, attention_win, max_length=50, padding_idx=0):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.attn = attention_mode
        self.attn_win = attention_win

        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=padding_idx)
        if attention_mode == 'no':
            self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
        elif attention_mode == 'global':
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
            self.attention = Attention(mode='global', hidden_dim=hidden_dim, src_length=max_length)
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, input, hidden, cell, encoder_output=None):

        if input.dim() == 1:
            input = input.unsqueeze(0)

        embedded = self.embedding(input)

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        if self.attn == 'global':
            output = self.attention(encoder_output, hidden[-1])

        hidden = torch.clamp(hidden, min=-50, max=50)
        cell = torch.clamp(cell, min=-50, max=50)

        output = self.fc_out(output)
        return output, hidden, cell
    

class Attention(nn.Module):
    def __init__(self, mode='global', hidden_dim=1000, src_length=50):
        super(Attention, self).__init__()
        self.mode = mode
        
        if mode == 'global':
            self.align = self.align_location
            self.W_a = nn.Linear(hidden_dim, src_length, bias=False)
            self.W_c = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        elif mode =='local':
            self.align = self.align_general
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, encoder_hidden, target_hidden):
        src_length, batch_size, hidden_dim = encoder_hidden.shape

        # calculate alignment score
        align = self.align(target_hidden, encoder_hidden) # (src_length, batch_size)
        align = align.T

        # calculate context vector by weighted averaging
        # Broadcast?
        context = torch.mul(encoder_hidden, align[:src_length].view(src_length, batch_size, 1)) # (src_length, batch_size, hidden_dim) * (src_length, batch_size) -> (src_length, batch_size, hidden_dim)
        context = torch.sum(context, dim=0) # -> (batch_size, hidden_dim)

        # concatenate context and hidden_state
        context = torch.cat((context, target_hidden), dim=1) # -> (batch_size, 2*hidden_dim)
        context = self.W_c(context)
        attention_hidden = F.tanh(context)

        return attention_hidden

    def align_general(self, trg_hidden, src_hidden):
        src_hidden = self.W_a(src_hidden)
        score = torch.dot(trg_hidden, src_hidden)
        align = F.softmax(score)
        return align

    def align_location(self, trg_hidden, src_hidden=None):
        # trg_hidden : (batch_size, hidden_dim)

        score = self.W_a(trg_hidden) # (batch_size, src_length)
        align = F.softmax(score, dim=1)
        return align
    