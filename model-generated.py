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
        pass
    
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, bidirectional, dropout, pad_idx):
        super(LSTMEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_dim * 2 if bidirectional else hidden_dim, num_heads=8)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [sent_len, batch_size]
        embedded = self.dropout(self.embedding(text))
        # embedded: [sent_len, batch_size, embed_dim]
        
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [sent_len, batch_size, hidden_dim * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # hidden: [batch_size, hidden_dim * num_directions]
        
        return outputs, hidden
    
class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout):
        super(LSTMDecoder, self).__init__()
        
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(embed_dim + hidden_dim * 3, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, context):
        # input: [batch_size]
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        # context: [batch_size, hidden_dim * num_directions]
        
        input = input.unsqueeze(0)
        # input: [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [1, batch_size, embed_dim]
        
        context = context.unsqueeze(0)
        # context: [1, batch_size, hidden_dim * num_directions]
        
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input: [1, batch_size, embed_dim + hidden_dim * num_directions]
        
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        # output: [1, batch_size, hidden_dim]
        
        output = output.squeeze(0)
        embedded = embedded.squeeze(0)
        context = context.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, embedded, context), dim=1))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden, cell