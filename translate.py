import torch
import argparse
from model import NMTRNN
import pickle
#import sys


parser = argparse.ArgumentParser()
parser.add_argument("--model_file", default="./models/ntrnn.pt")
parser.add_argument("--target_length", default=50)
parser.add_argument("--embed_dim", default=1000)
parser.add_argument("--hidden_dim", default=1000)
parser.add_argument("--num_layer", default=4)
parser.add_argument("--source_vocab", default="./models/ntrnn_src.pkl")
parser.add_argument("--target_vocab", default="./models/ntrnn_trg.pkl")
parser.add_argument("--reverse", "-r", type=bool, default=False)
parser.add_argument("--window", "-win", type=int, default=10)
parser.add_argument("--mode", "-m", default="beam_translate")
parser.add_argument("--beam_size", "-beam", type=int, default=5)
args = parser.parse_args()


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
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")

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
                  n_layers=args.num_layer, dropout=0.0, device=device, padding_idx=src_w2i['<pad>'])
    
    model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))

    print("Query: ")
    query = input()

    while(len(query)>0):
        tokenized_query = tokenize(query.strip().split(' '), src_w2i)
        tokenized_query = torch.tensor(tokenized_query).reshape(len(tokenized_query), 1)

        target_input = [trg_w2i['<sos>']] + [0 for i in range(args.target_length)]
        target_input = torch.tensor(target_input).reshape(len(target_input), 1)

        output = model.translate(tokenized_query, target_input, mode=args.mode, beam_size=args.beam_size, eos_token=trg_w2i['<eos>'])
        #output = output.argmax(2).squeeze()

        output_sentence = ""
        for id in output:
            output_sentence += trg_i2w[id.item()] + ' '
            if trg_i2w[id.item()] == '<eos>': break

        print(output_sentence)

        print("Query: ")
        query = input()