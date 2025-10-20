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
        self.decoder = LSTMDecoder(output_dim=output_size, embed_dim=embedding_dim, hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout, 
                                   attention_mode=attention_mode, attention_win=attention_win, input_feeding=input_feeding,
                                   max_length=max_length, padding_idx=padding_idx, device=device)
        self.init_weights()
        self.input_feeding = input_feeding
        self.device = device

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.1, 0.1)

    def forward(self, src, trg_input, src_lengths=None, mode='train'):
        trg_length, batch_size = trg_input.shape
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        encoder_output, (hidden, cell) = self.encoder(src, src_lengths)

        if self.attention_mode == 'no': encoder_output = None

        decoder_input = trg_input[0,:] # (batch_size)
        prev_context = None
        for t in range(trg_length):
            decoder_output, hidden, cell, attention_output = self.decoder(decoder_input, hidden, cell, encoder_output, prev_context)
            if self.input_feeding:
                prev_context = attention_output
            outputs[t] = decoder_output # (batch_size, trg_vocab_size)

            if t == trg_length-1: break
            if mode=='train' or mode=='test':
                decoder_input = trg_input[t+1,:]
            elif mode=='greedy_translate':
                decoder_input = decoder_output.argmax(1)
        
        return outputs

    def translate(self, src, trg_input, mode='beam_translate', beam_size=None, eos_token=None): # Note: beam search does not support batching
        self.eval()
        if mode=='greedy_translate' or mode=='greedy':
            with torch.no_grad():
                output = self.forward(src, trg_input, mode=mode)
                return torch.tensor(output.argmax(dim=-1))

        elif mode=='beam_translate' or mode=='beam':
            with torch.no_grad():
                trg_length, batch_size = trg_input.shape
                trg_vocab_size = self.decoder.output_dim

                encoder_output, (hidden, cell) = self.encoder(src, None)
                if self.attention_mode == 'no':
                    encoder_output = None

                # Initialize beam search
                beams = [(
                    [trg_input[0].item()],
                    0.0,
                    hidden.clone(),
                    cell.clone(),
                    None
                )]

                completed_sequences = []

                for t in range(1, trg_length):
                    candidates = []

                    for sequence, log_prob, h, c, context in beams:
                        decoder_input = torch.tensor([sequence[-1]], device=self.device)

                        decoder_output, new_h, new_c, attention_output = self.decoder(decoder_input, h, c, encoder_output, context)

                        # Fix: Handle the output shape properly
                        if decoder_output.dim() == 3:  # (1, batch_size, vocab_size)
                            decoder_output = decoder_output.squeeze(0)  # (batch_size, vocab_size)
                        
                        if decoder_output.dim() == 2 and decoder_output.size(0) == 1:  # (1, vocab_size)
                            decoder_output = decoder_output.squeeze(0)  # (vocab_size,)

                        # Get top k probabilities
                        log_probs = F.log_softmax(decoder_output, dim=-1)
                        top_log_probs, top_indices = log_probs.topk(beam_size)

                        # Create new candidates
                        for i in range(beam_size):
                            new_token = top_indices[i].item()
                            new_log_prob = log_prob + top_log_probs[i].item()
                            new_sequence = sequence + [new_token]
                            new_context = attention_output if self.input_feeding else None

                            candidate = (new_sequence, new_log_prob, new_h.clone(), new_c.clone(), new_context)
                            candidates.append(candidate)

                    # Sort candidates by log prob and keep top beam_size
                    candidates.sort(key=lambda x: x[1], reverse=True)

                    # Separate completed and ongoing sequences
                    new_beams = []
                    for candidate in candidates:
                        sequence, log_prob, h, c, context = candidate

                        if sequence[-1] == eos_token or len(sequence) >= trg_length:
                            completed_sequences.append(candidate)
                        else:
                            new_beams.append(candidate)

                        if len(new_beams) >= beam_size:
                            break
                    
                    beams = new_beams

                    if len(completed_sequences) >= beam_size or not beams:
                        break

                # Add remaining beams to completed sequences
                completed_sequences.extend(beams)

                # Sort by log prob and get best sequence
                completed_sequences.sort(key=lambda x: x[1], reverse=True)
                best_sequence = completed_sequences[0][0]
                if best_sequence[-1] == eos_token:
                    best_sequence = best_sequence[:-1]
                
                return torch.tensor(best_sequence)



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

        #hidden = torch.clamp(hidden, min=-50, max=50)
        #cell = torch.clamp(cell, min=-50, max=50)
        return output, (hidden, cell)
    
class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout, 
                 attention_mode, attention_win, input_feeding, max_length=50, padding_idx=0, device='cpu'):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.attn = attention_mode
        self.attn_win = attention_win
        self.input_feeding = input_feeding
        self.device = device

        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=padding_idx)
        if input_feeding: 
            embed_dim *= 2
        
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout)

        if attention_mode == 'global':
            self.attention = Attention(mode='global', hidden_dim=hidden_dim, src_length=max_length, device=device)
        elif attention_mode == 'local':
            self.attention = Attention(mode='local', hidden_dim=hidden_dim, src_length=attention_win, device=device)

        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, input, hidden, cell, encoder_output, prev_context):

        if input.dim() == 1:
            input = input.unsqueeze(0) # (batch_size) -> (length==1, batch_size)

        embedded = self.embedding(input)
        if self.input_feeding:
            if prev_context is not None:
                embedded = torch.cat((embedded, prev_context.unsqueeze(0)), dim=2) # (1, batch_size, embedded_size) + (1, batch_size, embedded_size)
            else:
                embedded = torch.cat((embedded, torch.zeros_like(embedded).to(self.device)), dim=2) # need to match the size during the first time step!
                

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        attention_output = None
        if self.attn == 'global':
            attention_output = self.attention(encoder_output, hidden[-1])
            output = attention_output
        elif self.attn == 'local':
            attention_output = self.attention(encoder_output, hidden[-1])
            output = attention_output

       #hidden = torch.clamp(hidden, min=-50, max=50)
       # cell = torch.clamp(cell, min=-50, max=50)

        output = self.fc_out(output)
        return output, hidden, cell, attention_output
    

class Attention(nn.Module):
    def __init__(self, mode='global', hidden_dim=1000, src_length=50, device='cpu'):
        super(Attention, self).__init__()
        self.mode = mode
        self.device = device
        
        if mode == 'global':
            self.align = self.align_global_location
            self.W_a = nn.Linear(hidden_dim, src_length, bias=False)
            self.W_c = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        elif mode =='local':
            self.align = self.align_local_general
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
        tanh_context = F.tanh(context)

        return tanh_context # (batch_size, hidden_dim)

    def align_local_general(self, trg_hidden, src_hidden): # local attention
        # trg_hidden : (batch_size, hidden_dim)
        # src_hidden : (src_length, batch_size, hidden_dim)

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

    def align_global_location(self, trg_hidden, src_hidden=None): # global attention
        # trg_hidden : (batch_size, hidden_dim)

        score = self.W_a(trg_hidden) # (batch_size, src_length)
        align = F.softmax(score, dim=1)
        return align, None
    

    ####### TODO ########
    def global_attention(self, trg_hidden, src_hidden):
        pass

    def local_attention(self, trg_hidden, src_hidden):
        pass
    
    def align_general(self, trg_hidden, src_hidden):
        pass

    def align_location(self, trg_hidden, src_hidden=None):
        pass
    
    def align_dot(self, trg_hidden, src_hidden):
        pass