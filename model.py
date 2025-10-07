import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NTRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, n_layers, dropout=None, device='cpu'):
        super(NTRNN, self).__init__()
        self.encoder = LSTMEncoder(vocab_size=input_size, embed_dim=embedding_dim, hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout)
        self.decoder = LSTMDecoder(output_dim=output_size, embed_dim=embedding_dim, hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout)
        self.init_weights()
        self.device = device

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.1, 0.1)

    def forward(self, src, trg_input, mode='train'):
        trg_length = trg_input.shape[0]
        batch_size = trg_input.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        encoder_output, (hidden, cell) = self.encoder(src)
        
        decoder_input = trg_input[0,:]
        for t in range(1, trg_length):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t-1] = decoder_output # shape (1, batch_size, trg_vocab_size)

            top1 = decoder_output.argmax(1)
            if mode=='train':
                decoder_input = trg_input[t,:]
            else:
                decoder_input = top1
        
        return outputs


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)

    def forward(self, text):

        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return output, (hidden, cell)
    
class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):

        if input.dim() == 1:
            input = input.unsqueeze(0)

        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc_out(output)
        return output, hidden, cell