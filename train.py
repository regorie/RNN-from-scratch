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
parser.add_argument("--train_src", "-trs", default='./datasets/datasets_small/train.10k.en')
parser.add_argument("--test_src", "-tes", default='./datasets/datasets_small/test.100.en')
parser.add_argument("--train_trg", "-trt", default='./datasets/datasets_small/train.10k.de')
parser.add_argument("--test_trg", "-tet", default='./datasets/datasets_small/test.100.de')
parser.add_argument("--val_src", "-vls", default='./datasets/datasets_small/valid.100.en')
parser.add_argument("--val_trg", "-vlt", default='./datasets/datasets_small/valid.100.de')
parser.add_argument("--save_path", "-sp", default='./models/nmtrnn.pt')
parser.add_argument("--vocab_path_base", "-vpb", default='./models/nmtrnn_')
parser.add_argument("--max_vocab", "-mv", type=int, default=50000)

parser.add_argument("--beam_size", '-beam', type=int, default=12)
parser.add_argument("--BLEU", default='yes')

parser.add_argument("--dropout", "-d", type=float, default=0.0)
parser.add_argument("--batch_size", "-b", type=int, default=128)
parser.add_argument("--test_batch_size", "-tb", type=int, default=100)
parser.add_argument("--epochs", "-ep", type=int, default=10)
parser.add_argument("--lr", "-lr", type=float, default=1.0)
parser.add_argument("--learning_rate_update_point", "-lrup", type=int, default=5)
parser.add_argument("--max_len", "-m", type=int, default=50)

parser.add_argument("--embed_dim", "-e", type=int, default=1000)
parser.add_argument("--hidden_dim", "-hd", type=int, default=1000)
parser.add_argument("--num_layer", "-nl", type=int, default=4)

parser.add_argument("--reverse", "-r", type=bool, default=False)
parser.add_argument("--window", "-win", type=int, default=10) # used for local attn
parser.add_argument("--attn_mode", "-attn", type=str, default='no') # no, global, local
parser.add_argument("--input_feeding", "-feed", type=bool, default=False)

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dict = vars(args)


    # set training params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("mps")

    # build vocab
    src_w2i, src_i2w = build_vocab(args.train_src, args.max_vocab)
    trg_w2i, trg_i2w = build_vocab(args.train_trg, args.max_vocab)

    # load data
    src_train, trg_train = load_data(args.train_src, args.train_trg, src_w2i=src_w2i, trg_w2i=trg_w2i, max_len=args.max_len, is_reverse=args.reverse, src_file=filtered_sentence_file['train_en'], trg=filtered_sentence_file['train_de'])
    src_test, trg_test = load_data(args.test_src, args.test_trg, src_w2i=src_w2i, trg_w2i=trg_w2i, max_len=args.max_len, is_reverse=args.reverse, src_file=filtered_sentence_file['test_en'], trg=filtered_sentence_file['test_de'])
    src_val, trg_val = load_data(args.val_src, args.val_trg, src_w2i=src_w2i, trg_w2i=trg_w2i, max_len=args.max_len, is_reverse=args.reverse, src_file=filtered_sentence_file['dev_en'], trg=filtered_sentence_file['dev_de'])

    train_dataset = TextDataset(src_sentences=src_train, trg_sentences=trg_train, sos=trg_w2i['<sos>'], eos=trg_w2i['<eos>'])
    train_loader = get_data_loader(train_dataset, batch_size=args.batch_size, pad_idx=src_w2i['<pad>'], shuffle=True, drop_last=True)
    test_dataset = TextDataset(src_sentences=src_test, trg_sentences=trg_test, sos=trg_w2i['<sos>'], eos=trg_w2i['<eos>'])
    test_loader = get_data_loader(test_dataset, batch_size=args.test_batch_size, pad_idx=src_w2i['<pad>'], shuffle=False, drop_last=False)
    val_dataset = TextDataset(src_sentences=src_val, trg_sentences=trg_val, sos=trg_w2i['<sos>'], eos=trg_w2i['<eos>'])
    val_loader = get_data_loader(val_dataset, batch_size=args.test_batch_size, pad_idx=src_w2i['<pad>'], shuffle=False, drop_last=False)

    # setup model
    model = NMTRNN(input_size=len(src_w2i), embedding_dim=args.embed_dim, hidden_size=args.hidden_dim, output_size=len(trg_w2i), 
                  n_layers=args.num_layer, dropout=args.dropout, input_feeding=args.input_feeding, attention_mode=args.attn_mode, attention_win=args.window, max_length=args.max_len,
                    device=device, padding_idx=src_w2i['<pad>'])
    with torch.no_grad():
        model.encoder.embedding.weight[0] = torch.zeros(args.embed_dim)
        model.decoder.embedding.weight[0] = torch.zeros(args.embed_dim)
    model = model.to(device)

    # learning rate scheduler, optimizer, loss function
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_w2i['<pad>'])

    # setup trainer
    trainer = Trainer(model, train_loader, test_loader, val_loader, optimizer, criterion, device, learning_rate_update_point=args.learning_rate_update_point)

    # training loop
    test_interval = (train_dataset.__len__() // args.batch_size) // 20 # test loss recorded 20 times per epoch
    start_time = datetime.now()
    trainer.train(args.epochs, test_interval)
    end_time = datetime.now()

    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training took: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")


    # test model
    trainer.model.load_state_dict(trainer.best_model_state)
    test_loss = trainer.evaluate()
    logs_dict["final_test_loss"] = test_loss

    print("Final test loss: ", test_loss)

    # save model
    torch.save(trainer.model.state_dict(), args.save_path)

    # save vocab
    source_vocab_file = args.vocab_path_base + "src.pkl"
    target_vocab_file = args.vocab_path_base + "trg.pkl"

    src_vocab = {'w2i': src_w2i,
                 'i2w': src_i2w}
    trg_vocab = {'w2i': trg_w2i,
                 'i2w': trg_i2w}
    with open(source_vocab_file, "wb") as f:
        pickle.dump(src_vocab, f)
    with open(target_vocab_file, "wb") as f:
        pickle.dump(trg_vocab, f)

    logs_dict["test_loss"] = trainer.loss_list
    logs_dict["src_vocab_file"] = source_vocab_file
    logs_dict["trg_vocab_file"] = target_vocab_file
    
    with open(f'./logs/training_log_{timestamp}.json', 'w') as f:
        json.dump(logs_dict, f, indent=2)

    print("Saving complete...")

    # test model BLEU score
    del train_loader, test_loader, val_loader
    del train_dataset, val_dataset

    total_bleu_score = 0
    if args.BLEU == 'yes':
        print("Calculating BLEU... ")
        bleu_loader = get_data_loader(test_dataset, batch_size=1, pad_idx=src_w2i['<pad>'], shuffle=False, drop_last=False)

        for batch in tqdm(bleu_loader):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            output = trainer.model.translate(source, target, mode='beam_translate', beam_size=args.beam_size, eos_token=trg_w2i['<eos>']) # (length)

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

        logs_dict["BLEU_score"] = total_bleu_score
    
        with open(f'./logs/training_log_{timestamp}_with_bleu.json', 'w') as f:
            json.dump(logs_dict, f, indent=2)
