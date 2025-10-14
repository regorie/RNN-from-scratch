import torch
import argparse
from model import NMTRNN
from dataset import load_data, get_data_loader, TextDataset
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--train_src", "-trs", default='./datasets/datasets_small/train.10k.en')
parser.add_argument("--test_src", "-tes", default='./datasets/datasets_small/test.100.en')
parser.add_argument("--train_trg", "-trt", default='./datasets/datasets_small/train.10k.de')
parser.add_argument("--test_trg", "-tet", default='./datasets/datasets_small/test.100.de')
parser.add_argument("--val_src", "-vls", default='./datasets/datasets_small/valid.100.en')
parser.add_argument("--val_trg", "-vlt", default='./datasets/datasets_small/valid.100.de')
parser.add_argument("--save_path", "-sp", default='./models/nmtrnn.pt')
parser.add_argument("--vocab_path_base", "-vpb", default='./models/nmtrnn_')

parser.add_argument("--beam_size", '-beam', type=int, default=12)

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

#torch.autograd.set_detect_anomaly(True)

if __name__=='__main__':
    # set training params
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps")

    # load datas
    src_train, trg_train, src_vocab, trg_vocab = load_data(args.train_src, args.train_trg, max_len=args.max_len, is_reverse=args.reverse)
    vocab_tuple = (src_vocab[0], trg_vocab[0])
    src_test, trg_test, _, _ = load_data(args.test_src, args.test_trg, max_len=args.max_len, is_reverse=args.reverse, vocab=vocab_tuple)
    src_val, trg_val, _, _ = load_data(args.val_src, args.val_trg, max_len=args.max_len, is_reverse=args.reverse, vocab=vocab_tuple)
    del _
    train_dataset = TextDataset(src_sentences=src_train, trg_sentences=trg_train, src_vocab=src_vocab, trg_vocab=trg_vocab)
    train_loader = get_data_loader(train_dataset, batch_size=args.batch_size, pad_idx=src_vocab[0]['<pad>'], shuffle=True, drop_last=True)
    test_dataset = TextDataset(src_sentences=src_test, trg_sentences=trg_test, src_vocab=src_vocab, trg_vocab=trg_vocab)
    test_loader = get_data_loader(test_dataset, batch_size=args.test_batch_size, pad_idx=src_vocab[0]['<pad>'], shuffle=False, drop_last=False)
    val_dataset = TextDataset(src_sentences=src_val, trg_sentences=trg_val, src_vocab=src_vocab, trg_vocab=trg_vocab)
    val_loader = get_data_loader(val_dataset, batch_size=args.test_batch_size, pad_idx=src_vocab[0]['<pad>'], shuffle=False, drop_last=False)

    # setup model
    model = NMTRNN(input_size=len(src_vocab[0]), embedding_dim=args.embed_dim, hidden_size=args.hidden_dim, output_size=len(trg_vocab[0]), 
                  n_layers=args.num_layer, dropout=args.dropout, input_feeding=args.input_feeding, attention_mode=args.attn_mode, attention_win=args.window, max_length=args.max_len,
                    device=device, padding_idx=src_vocab[0]['<pad>'])
    with torch.no_grad():
        model.encoder.embedding.weight[0] = torch.zeros(args.embed_dim)
        model.decoder.embedding.weight[0] = torch.zeros(args.embed_dim)
    model = model.to(device)

    # learning rate scheduler, optimizer, loss function
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab[0]['<pad>'])

    # setup trainer
    trainer = Trainer(model, train_loader, test_loader, val_loader, optimizer, criterion, device, learning_rate_update_point=args.learning_rate_update_point)

    # training loop
    trainer.train(args.epochs)

    # test model
    trainer.model.load_state_dict(trainer.best_model_state)
    test_loss = trainer.evaluate()

    print("Final test loss: ", test_loss)

    # save model
    torch.save(trainer.model.state_dict(), args.save_path)

    # save vocab
    source_vocab_file = args.vocab_path_base + "src.pkl"
    target_vocab_file = args.vocab_path_base + "trg.pkl"

    with open(source_vocab_file, "wb") as f:
        pickle.dump(src_vocab, f)
    with open(target_vocab_file, "wb") as f:
        pickle.dump(trg_vocab, f)

    print("Saving complete... Calculating BLEU score")

    # test model BLEU score
    total_bleu_score = 0
    for batch in test_loader:
        for b in tqdm(range(len(batch["source"][0]))):
            source = batch["source"][:,b].to(device).reshape(-1,1)
            target = batch["target"][:,b].to(device).reshape(-1,1)

            output = trainer.model.translate(source, target, mode='beam_translate', beam_size=args.beam_size, eos_token=trg_vocab[0]['<eos>']) # (length)

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