import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NTRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, n_layers, dropout=None, device='cpu', padding_idx=0):
        super(NTRNN, self).__init__()
        self.encoder = LSTMEncoder(vocab_size=input_size, embed_dim=embedding_dim, hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout, padding_idx=padding_idx)
        self.decoder = LSTMDecoder(output_dim=output_size, embed_dim=embedding_dim, hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout, padding_idx=padding_idx)
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
        for t in range(trg_length):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = decoder_output.squeeze(0) # shape (1, batch_size, trg_vocab_size)

            if t == trg_length-1: break
            if mode=='train':
                decoder_input = trg_input[t+1,:]
            else:
                decoder_input = decoder_output.argmax(2).squeeze(0)
        
        return outputs

    def predict(self, src, trg_input):
        self.eval()
        with torch.no_grad():
            output = self.forward(src, trg_input, mode='predict')
        return output

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, padding_idx=0):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)

    def forward(self, text):

        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.clamp(hidden, min=-50, max=50)
        cell = torch.clamp(cell, min=-50, max=50)
        return output, (hidden, cell)
    
class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout, padding_idx=0):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=padding_idx)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, input, hidden, cell):

        if input.dim() == 1:
            input = input.unsqueeze(0)

        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        hidden = torch.clamp(hidden, min=-50, max=50)
        cell = torch.clamp(cell, min=-50, max=50)

        output = self.fc_out(output)
        return output, hidden, cell