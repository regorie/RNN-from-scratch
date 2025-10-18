import torch
import argparse
from model import NMTRNN
from dataset import load_data, get_data_loader, build_vocab, TextDataset
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

parser = argparse.ArgumentParser()


parser.add_argument("--beam_size", '-beam', type=int, default=12)


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

parser.add_argument("--test_src", "-tes", default='./datasets/wmt-commoncrawl/test_2015_en.txt')
parser.add_argument("--test_trg", "-tet", default='./datasets/wmt-commoncrawl/test_2015_de.txt')

args = parser.parse_args()

filtered_sentence_file={
    'train_en': None,
    'train_de': None,
    'test_en': None,
    'test_de': None,
    'dev_en': None,
    'dev_de': None,
}

if __name__=='__main__':

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps")


    # load vocab
    with open(args.src_vocab_file, "rb") as sf, open(args.trg_vocab_file, "rb") as tf:
        src_vocab = pickle.load(sf)
        trg_vocab = pickle.load(tf)

        src_w2i = src_vocab["w2i"]
        src_i2w = src_vocab["i2w"]
        trg_w2i = trg_vocab["w2i"]
        trg_i2w = trg_vocab["i2w"]

    # load model
    model = model = NMTRNN(input_size=len(src_w2i), embedding_dim=args.embed_dim, hidden_size=args.hidden_dim, output_size=len(trg_w2i), 
                  n_layers=args.num_layer, dropout=args.dropout, input_feeding=args.input_feeding, attention_mode=args.attn_mode, attention_win=args.window, max_length=args.max_len,
                    device=device, padding_idx=src_w2i['<pad>'])
    
    model.load_state_dict(torch.load(args.model_file, weights_only=True, map_location=torch.device('cpu')))
    model.to(device)

    # load data
    src_test, trg_test = load_data(args.test_src, args.test_trg, src_w2i=src_w2i, trg_w2i=trg_w2i, max_len=args.max_len, is_reverse=args.reverse, src_file=filtered_sentence_file['test_en'], trg_file=filtered_sentence_file['test_de'])
    test_dataset = TextDataset(src_sentences=src_test, trg_sentences=trg_test, sos=trg_w2i['<sos>'], eos=trg_w2i['<eos>'])
    bleu_loader = get_data_loader(test_dataset, batch_size=1, pad_idx=src_w2i['<pad>'], shuffle=False, drop_last=False)

    print("Calculating BLEU... ")

    total_bleu_score = 0
    for batch in tqdm(bleu_loader):
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        output = model.translate(source, target, mode='beam_translate', beam_size=args.beam_size, eos_token=trg_w2i['<eos>']) # (length)

        p_i = []
        for n in range(1, 5):
            
            # transfer to ngram
            predicted_n_gram_list = []
            target_n_gram_list = []
            for i in range(len(output)-n+1):
                predicted_n_gram_list.append(tuple(output[i:i+n]))
            for i in range(len(target)-n+1):
                target_n_gram_list.append(tuple(target[i:i+n]))
            
            target_counter = Counter(target_n_gram_list)
            predicted_counter = Counter(predicted_n_gram_list)

            common_keys = set(predicted_counter.keys()) & set(target_counter.keys())
            correct_counter = predicted_counter & target_counter
            correct = sum(correct_counter.values())
            total = sum(predicted_counter.values())

            if total > 0 and correct > 0:
                p_i.append(correct/total)

        if p_i:
            bp = min(1.0, np.exp(1 - len(output)/len(target)))
            bleu_score = bp * np.average(np.log(p_i))
            total_bleu_score += bleu_score
    
    print("BLEU score: ", total_bleu_score)

