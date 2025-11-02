import torch
import torch.nn.functional as F
import argparse
from model import NMTRNN
import pickle
from tqdm import tqdm
#import sys


parser = argparse.ArgumentParser()
parser.add_argument("--model_file", default="./models/base.pt")
parser.add_argument("--source_vocab", default="./models/base_src.pkl")
parser.add_argument("--target_vocab", default="./models/base_trg.pkl")
parser.add_argument("--source_file", default=None)
parser.add_argument("--output_file", default=None, help="Path to write translated sentences (one per line)")

parser.add_argument("--embed_dim", default=1000)
parser.add_argument("--hidden_dim", default=1000)
parser.add_argument("--num_layer", default=4)
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--attn_mode", default='no')
parser.add_argument("--input_feeding", default=False, type=bool)
parser.add_argument("--max_len", default=50)

parser.add_argument("--reverse", "-r", type=bool, default=False)
parser.add_argument("--window", "-win", type=int, default=10)

parser.add_argument("--mode", "-m", default="beam_translate")
parser.add_argument("--beam_size", "-beam", type=int, default=6)
parser.add_argument("--target_length", default=50)

parser.add_argument("--unk_replacement", "-unk", default=False, type=bool)

args = parser.parse_args()

if args.source_file is not None:
    input_file = args.source_file

def tokenize(query, w2i):
    result = []
    for word in query:
        if word in w2i:
            result.append(w2i[word])
        else:
            result.append(w2i['<unk>'])

    return result

class BeamPositionTracker:
    def __init__(self):
        self.all_positions = {}  # beam_id -> list of positions
        self.current_beam_id = None
        
    def track_beam_positions(self, model, beam_id):
        """Track positions for a specific beam"""
        self.current_beam_id = beam_id
        if beam_id not in self.all_positions:
            self.all_positions[beam_id] = []
            
        if hasattr(model.decoder, 'attention') and model.decoder.attention.mode == 'local':
            # Attention.forward uses self.align (set in Attention.__init__), so patch .align
            original_align = model.decoder.attention.align
            
            def position_capturing_align(trg_hidden, src_hidden):
                align, position_t = original_align(trg_hidden, src_hidden)
                if position_t is not None and self.current_beam_id is not None:
                    # store a detached cpu copy for later replacement
                    self.all_positions[self.current_beam_id].append(position_t.detach().cpu())
                return align, position_t
            
            model.decoder.attention.align = position_capturing_align
            return original_align
        return None
    
    def restore_original(self, model, original_align):
        """Restore original alignment function"""
        if original_align is not None:
            model.decoder.attention.align = original_align
    
    def get_beam_positions(self, beam_id):
        return self.all_positions.get(beam_id, [])
    
    def clear(self):
        self.all_positions = {}
        self.current_beam_id = None

def unk_replacement(source_words, target_words, source_positions, src_w2i, trg_w2i, trg_i2w):
    """
    Replace <unk> tokens in target with corresponding source words based on attention positions
    """
    if not source_positions or len(source_positions) == 0:
        return target_words
    
    replaced_words = target_words.copy()
    unk_token = '<unk>'
    
    for i, word in enumerate(target_words):
        if word == unk_token and i < len(source_positions):
            # Get the attention position for this target word
            pos_tensor = source_positions[i]
            if pos_tensor.numel() > 0:
                # Convert position to integer (round to nearest)
                pos = int(torch.round(pos_tensor).item())
                # Make sure position is within source bounds
                if 0 <= pos < len(source_words):
                    # Replace with source word at attended position
                    replaced_words[i] = source_words[pos]
    
    return replaced_words

def translate_with_unk_replacement(model, src_tensor, target_tensor, source_words, src_w2i, trg_w2i, trg_i2w, mode='beam', beam_size=6, eos_token=None):
    """
    Translate with unknown word replacement using attention positions
    """
    if mode == 'greedy' or model.decoder.attn != 'local':
        # Fallback to regular translation for greedy or non-local attention
        output_ids = model.translate(src_tensor, target_tensor, mode=mode, beam_size=beam_size, eos_token=eos_token)
        words = []
        for tid in output_ids:
            word = trg_i2w[tid.item()]
            if word == '<eos>':
                break
            if word == '<sos>':
                continue
            words.append(word)
        return words
    
    # For beam search with local attention and UNK replacement
    tracker = BeamPositionTracker()
    
    with torch.no_grad():
        trg_length, batch_size = target_tensor.shape
        trg_vocab_size = model.decoder.output_dim

        encoder_output, (hidden, cell) = model.encoder(src_tensor, None)
        if model.attention_mode == 'no':
            encoder_output = None

        # Initialize beam search with position tracking
        beams = [(
            [target_tensor[0].item()],
            0.0,
            hidden.clone(),
            cell.clone(),
            None,
            0  # beam_id
        )]

        completed_sequences = []
        beam_id_counter = 0

        for t in range(1, trg_length):
            candidates = []
            
            for sequence, log_prob, h, c, context, beam_id in beams:
                # Track positions for this beam
                original_align = tracker.track_beam_positions(model, beam_id)
                
                try:
                    decoder_input = torch.tensor([sequence[-1]], device=model.device)
                    decoder_output, new_h, new_c, attention_output = model.decoder(decoder_input, h, c, encoder_output, context)

                    # Handle output shape
                    if decoder_output.dim() == 3:
                        decoder_output = decoder_output.squeeze(0)
                    if decoder_output.dim() == 2 and decoder_output.size(0) == 1:
                        decoder_output = decoder_output.squeeze(0)

                    # Get top k probabilities
                    log_probs = F.log_softmax(decoder_output, dim=-1)
                    top_log_probs, top_indices = log_probs.topk(beam_size)

                    # Create new candidates
                    for i in range(beam_size):
                        new_token = top_indices[i].item()
                        new_log_prob = log_prob + top_log_probs[i].item()
                        new_sequence = sequence + [new_token]
                        new_context = attention_output if model.input_feeding else None
                        new_beam_id = beam_id_counter
                        beam_id_counter += 1

                        # Copy positions from parent beam
                        if beam_id in tracker.all_positions:
                            tracker.all_positions[new_beam_id] = tracker.all_positions[beam_id].copy()

                        candidate = (new_sequence, new_log_prob, new_h.clone(), new_c.clone(), new_context, new_beam_id)
                        candidates.append(candidate)
                        
                finally:
                    # Restore original alignment function
                    tracker.restore_original(model, original_align)

            # Sort candidates and select top beams
            candidates.sort(key=lambda x: x[1], reverse=True)

            new_beams = []
            for candidate in candidates:
                sequence, log_prob, h, c, context, beam_id = candidate

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
        best_sequence, _, _, _, _, best_beam_id = completed_sequences[0]
        
        # Get positions for the best beam
        best_positions = tracker.get_beam_positions(best_beam_id)
        
        # Convert sequence to words
        words = []
        for tid in best_sequence:
            if tid == eos_token:
                break
            if tid == trg_w2i['<sos>']:
                continue
            words.append(trg_i2w[tid])
        
        # Apply UNK replacement
        if args.unk_replacement and best_positions:
            words = unk_replacement(source_words, words, best_positions, src_w2i, trg_w2i, trg_i2w)
        
        return words

if __name__=='__main__':
    # set training params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load vocab
    with open(args.source_vocab, "rb") as sf,\
        open(args.target_vocab, "rb") as tf:
        src_vocab = pickle.load(sf)
        src_w2i, src_i2w = src_vocab["w2i"], src_vocab["i2w"]
        trg_vocab = pickle.load(tf)
        trg_w2i, trg_i2w = trg_vocab["w2i"], trg_vocab["i2w"]
        del trg_vocab, src_vocab
    # setup model
    model = NMTRNN(input_size=len(src_w2i), embedding_dim=args.embed_dim, hidden_size=args.hidden_dim, output_size=len(trg_w2i), 
                  n_layers=args.num_layer, dropout=args.dropout, input_feeding=args.input_feeding, attention_mode=args.attn_mode, attention_win=args.window, max_length=args.max_len,
                    device=device, padding_idx=src_w2i['<pad>'])
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.to(device)
    model.eval()

    # File-to-file translation mode
    if args.source_file is not None:
        output_path = args.output_file or (args.source_file + ".de.txt")
        with open(args.source_file, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
            with torch.no_grad():
                for line in tqdm(fin):
                    line = line.rstrip('\n')
                    if len(line.strip()) == 0:
                        # Preserve alignment: write empty line
                        fout.write("\n")
                        continue

                    source_words = line.strip().split(' ')
                    token_ids = tokenize(source_words, src_w2i)

                    if len(token_ids) > args.max_len:
                        # skip this sentence
                        fout.write("\n")
                        continue

                    if args.reverse:
                        token_ids = token_ids[::-1]
                        source_words = source_words[::-1]
                    
                    src_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).reshape(len(token_ids), 1)

                    target_input = [trg_w2i['<sos>']] + [0 for _ in range(int(args.target_length))]
                    target_tensor = torch.tensor(target_input, dtype=torch.long, device=device).reshape(len(target_input), 1)

                    if args.unk_replacement:
                        words = translate_with_unk_replacement(
                            model, src_tensor, target_tensor, source_words,
                            src_w2i, trg_w2i, trg_i2w, mode=args.mode, 
                            beam_size=args.beam_size, eos_token=trg_w2i['<eos>']
                        )
                    else:
                        output_ids = model.translate(src_tensor, target_tensor, mode=args.mode, beam_size=args.beam_size, eos_token=trg_w2i['<eos>'])
                        # Build output sentence; stop at <eos> and do not include it
                        words = []
                        for tid in output_ids:
                            word = trg_i2w[tid.item()]
                            if word == '<eos>':
                                break
                            if word == '<sos>':
                                continue
                            words.append(word)

                    fout.write((' '.join(words)).strip() + "\n")

        print(f"Wrote translations to: {output_path}")

    else:
        # Interactive mode
        print("Query: ")
        query = input()

        while(len(query)>0):
            source_words = query.strip().split(' ')
            tokenized_query = tokenize(source_words, src_w2i)
            if args.reverse:
                tokenized_query = tokenized_query[::-1]
                source_words = source_words[::-1]
            tokenized_query = torch.tensor(tokenized_query, dtype=torch.long, device=device).reshape(len(tokenized_query), 1)

            target_input = [trg_w2i['<sos>']] + [0 for i in range(int(args.target_length))]
            target_input = torch.tensor(target_input, dtype=torch.long, device=device).reshape(len(target_input), 1)

            with torch.no_grad():
                if args.unk_replacement:
                    words = translate_with_unk_replacement(
                        model, tokenized_query, target_input, source_words,
                        src_w2i, trg_w2i, trg_i2w, mode=args.mode, 
                        beam_size=args.beam_size, eos_token=trg_w2i['<eos>']
                    )
                    output_sentence = ' '.join(words)
                else:
                    output = model.translate(tokenized_query, target_input, mode=args.mode, beam_size=args.beam_size, eos_token=trg_w2i['<eos>'])
                    output_sentence = ""
                    for id in output:
                        output_sentence += trg_i2w[id.item()] + ' '
                        if trg_i2w[id.item()] == '<eos>': break

            print(output_sentence)

            print("Query: ")
            query = input()