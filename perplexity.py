import torch
import torch.nn.functional as F
import argparse
from model import NMTRNN
from dataset import load_data, get_data_loader, TextDataset
import pickle
import numpy as np
from tqdm import tqdm
import math

parser = argparse.ArgumentParser()

parser.add_argument("--dropout", "-d", type=float, default=0.0)
parser.add_argument("--max_len", "-m", type=int, default=50)

parser.add_argument("--embed_dim", "-e", type=int, default=1000)
parser.add_argument("--hidden_dim", "-hd", type=int, default=1000)
parser.add_argument("--num_layer", "-nl", type=int, default=4)

parser.add_argument("--reverse", "-r", type=bool, default=False)
parser.add_argument("--window", "-win", type=int, default=10) # used for local attn
parser.add_argument("--attn_mode", "-attn", type=str, default='no') # no, global, local, base
parser.add_argument("--input_feeding", "-feed", type=bool, default=False)

parser.add_argument("--model_file", "-model", default='./models/base.pt')
parser.add_argument("--src_vocab_file", "-src", default='./models/base_src.pkl')
parser.add_argument("--trg_vocab_file", "-trg", default='./models/base_trg.pkl')

parser.add_argument("--test_src", "-tes", default='./datasets/wmt-commoncrawl/test_2014_en.txt')
parser.add_argument("--test_trg", "-tet", default='./datasets/wmt-commoncrawl/test_2014_de.txt')

parser.add_argument("--batch_size", "-bs", type=int, default=1)

args = parser.parse_args()

filtered_sentence_file = {
    'train_en': None,
    'train_de': None,
    'test_en': None,
    'test_de': None,
    'dev_en': None,
    'dev_de': None,
}

def calculate_perplexity(model, data_loader, device, pad_idx, eos_idx):
    """
    Calculate perplexity of the model on the given dataset.
    Perplexity = exp(average negative log-likelihood per word)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    skipped_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating perplexity"):
            source = batch["source"].to(device)  # (src_len, batch_size)
            target = batch["target"].to(device)  # (trg_len, batch_size)
            
            # Skip if target is too short (just <sos> and <eos>)
            if target.size(0) <= 2:
                skipped_batches += 1
                continue
                
            # Prepare decoder input (target without last token)
            decoder_input = target[:-1, :]  # (trg_len-1, batch_size)
            # Prepare target output (target without first token) 
            target_output = target[1:, :]   # (trg_len-1, batch_size)
            
            try:
                # Forward pass through the model
                outputs = model(source, decoder_input, src_lengths=batch['src_lengths'], mode='train')  # (trg_len-1, batch_size, vocab_size)
                
                # Reshape for loss calculation
                outputs = outputs.reshape(-1, outputs.size(-1))  # (trg_len-1 * batch_size, vocab_size)
                target_output = target_output.reshape(-1)        # (trg_len-1 * batch_size)
                
                # Calculate cross-entropy loss
                loss = F.cross_entropy(outputs, target_output, ignore_index=pad_idx, reduction='sum')
                
                # Count non-padding tokens
                non_pad_tokens = (target_output != pad_idx).sum().item()
                
                # Skip if loss is invalid
                if torch.isnan(loss) or torch.isinf(loss) or non_pad_tokens == 0:
                    continue
                
                total_loss += loss.item()
                total_tokens += non_pad_tokens
                batch_count += 1
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue
    
    print(f"Processed {batch_count} batches")
    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batches (too short)")
    print(f"Total tokens: {total_tokens}")
    
    # Calculate average negative log-likelihood per token
    avg_nll = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    # Perplexity is exp(average negative log-likelihood)
    if avg_nll == float('inf') or avg_nll > 100:  # Cap very large values to prevent overflow
        perplexity = float('inf')
    else:
        perplexity = math.exp(avg_nll)
    
    return perplexity, avg_nll

if __name__ == '__main__':
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps")
    
    print(f"Using device: {device}")
    
    # Load vocabulary
    with open(args.src_vocab_file, "rb") as sf, open(args.trg_vocab_file, "rb") as tf:
        src_vocab = pickle.load(sf)
        trg_vocab = pickle.load(tf)
        
        src_w2i = src_vocab["w2i"]
        src_i2w = src_vocab["i2w"]
        trg_w2i = trg_vocab["w2i"]
        trg_i2w = trg_vocab["i2w"]
    
    print(f"Source vocabulary size: {len(src_w2i)}")
    print(f"Target vocabulary size: {len(trg_w2i)}")
    
    # Load model
    model = NMTRNN(
        input_size=len(src_w2i), 
        embedding_dim=args.embed_dim, 
        hidden_size=args.hidden_dim, 
        output_size=len(trg_w2i), 
        n_layers=args.num_layer, 
        dropout=args.dropout, 
        input_feeding=args.input_feeding, 
        attention_mode=args.attn_mode, 
        attention_win=args.window, 
        max_length=args.max_len,
        device=device, 
        padding_idx=src_w2i['<pad>']
    )
    
    model.load_state_dict(torch.load(args.model_file, weights_only=True, map_location=torch.device('cpu')))
    model.to(device)
    
    print("Model loaded successfully")
    
    # Load test data
    src_test, trg_test = load_data(
        args.test_src, args.test_trg, 
        src_w2i=src_w2i, trg_w2i=trg_w2i, 
        max_len=args.max_len, is_reverse=args.reverse, 
        src_file=filtered_sentence_file['test_en'], 
        trg_file=filtered_sentence_file['test_de']
    )
    
    test_dataset = TextDataset(
        src_sentences=src_test, 
        trg_sentences=trg_test, 
        sos=trg_w2i['<sos>'], 
        eos=trg_w2i['<eos>']
    )
    
    test_loader = get_data_loader(
        test_dataset, 
        batch_size=args.batch_size, 
        pad_idx=src_w2i['<pad>'], 
        shuffle=False, 
        drop_last=False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of batches: {len(test_loader)}")
    
    # Calculate perplexity
    print("\nCalculating perplexity...")
    perplexity, avg_nll = calculate_perplexity(
        model, test_loader, device, 
        pad_idx=trg_w2i['<pad>'], 
        eos_idx=trg_w2i['<eos>']
    )
    
    print(f"\nResults:")
    print(f"Average Negative Log-Likelihood: {avg_nll:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    # Also calculate bits per word (optional)
    bits_per_word = avg_nll / math.log(2)
    print(f"Bits per word: {bits_per_word:.4f}")