import torch
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

                    token_ids = tokenize(line.strip().split(' '), src_w2i)

                    if len(token_ids) > args.max_len:
                        # skip this sentence
                        fout.write("\n")
                        continue

                    if args.reverse:
                        token_ids = token_ids[::-1]
                    src_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).reshape(len(token_ids), 1)

                    target_input = [trg_w2i['<sos>']] + [0 for _ in range(int(args.target_length))]
                    target_tensor = torch.tensor(target_input, dtype=torch.long, device=device).reshape(len(target_input), 1)

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
            tokenized_query = tokenize(query.strip().split(' '), src_w2i)
            if args.reverse:
                tokenized_query = tokenized_query[::-1]
            tokenized_query = torch.tensor(tokenized_query, dtype=torch.long, device=device).reshape(len(tokenized_query), 1)

            target_input = [trg_w2i['<sos>']] + [0 for i in range(int(args.target_length))]
            target_input = torch.tensor(target_input, dtype=torch.long, device=device).reshape(len(target_input), 1)

            with torch.no_grad():
                output = model.translate(tokenized_query, target_input, mode=args.mode, beam_size=args.beam_size, eos_token=trg_w2i['<eos>'])
                
            output_sentence = ""
            for id in output:
                output_sentence += trg_i2w[id.item()] + ' '
                if trg_i2w[id.item()] == '<eos>': break

            print(output_sentence)

            print("Query: ")
            query = input()