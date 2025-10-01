import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NTRNN(nn.Module):
    def __init__(self, encoder, decoder):
        super(NTRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, text):
        encoder_outputs, (hidden, cell) = self.encoder(text)
        context = hidden[-1]
        
        decoder_input = text[-1, :]  # Start decoding from the last input token
        output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
        return output, context


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=1000, hidden_dim=1000, n_layers=4):
        super(LSTMEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers)

    def forward(self, text):
        """
        text: [sent_len, batch_size]
        """
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return output, (hidden, cell)
    
class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim=1000, hidden_dim=1000, n_layers=4):
        super(LSTMDecoder, self).__init__()

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers)

    def forward(self, input, hidden, cell):
        """
        input: [batch_size, 1]
        hidden: [n_layers, batch_size, hidden_dim]
        cell: [n_layers, batch_size, hidden_dim]
        """
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return output, (hidden, cell)