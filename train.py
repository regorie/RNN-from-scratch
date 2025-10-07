import torch
import argparse
from model import NTRNN
from dataset import load_data, get_data_loader, TextDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--train_src", "-trs", default='./datasets/train.10k.en')
parser.add_argument("--test_src", "-tes", default='./datasets/test.100.en')
parser.add_argument("--train_trg", "-trt", default='./datasets/train.10k.de')
parser.add_argument("--test_trg", "-tet", default='./datasets/test.100.de')
parser.add_argument("--val_src", "-vls", default='./datasets/valid.100.en')
parser.add_argument("--val_trg", "-vlt", default='./datasets/valid.100.de')
parser.add_argument("--reverse", "-r", default=False)
parser.add_argument("--window", "-win", default=10)
parser.add_argument("--embed_dim", "-e", type=int, default=1000)
parser.add_argument("--hidden_dim", "-hd", type=int, default=1000)
parser.add_argument("--dropout", "-d", type=float, default=0.0)
parser.add_argument("--batch_size", "-b", type=int, default=128)
parser.add_argument("--epochs", "-ep", type=int, default=10)
parser.add_argument("--lr", "-lr", type=float, default=1.0)
parser.add_argument("--max_len", "-m", type=int, default=50)
parser.add_argument("--save_path", "-s", default='./models/ntrnn.pkl')
args = parser.parse_args()


if __name__=='__main__':
    # set training params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    max_len = args.max_len
    save_path = args.save_path

    # load datas
    src_train, trg_train, src_vocab, trg_vocab = load_data(args.train_src, args.train_trg, max_len=max_len, is_reverse=args.reverse)
    src_test, trg_test, _, _ = load_data(args.test_src, args.test_trg, max_len=max_len, is_reverse=args.reverse)
    src_val, trg_val, _, _ = load_data(args.val_src, args.val_trg, max_len=max_len, is_reverse=args.reverse)

    train_dataset = TextDataset(src_sentences=src_train, trg_sentences=trg_train, src_vocab=src_vocab, trg_vocab=trg_vocab)
    train_loader = get_data_loader(train_dataset, batch_size=batch_size, pad_idx=src_vocab[0]['<pad>'], shuffle=True)
    test_dataset = TextDataset(src_sentences=src_test, trg_sentences=trg_test, src_vocab=src_vocab, trg_vocab=trg_vocab)
    test_loader = get_data_loader(test_dataset, batch_size=batch_size, pad_idx=src_vocab[0]['<pad>'], shuffle=False)
    val_dataset = TextDataset(src_sentences=src_val, trg_sentences=trg_val, src_vocab=src_vocab, trg_vocab=trg_vocab)
    val_loader = get_data_loader(val_dataset, batch_size=batch_size, pad_idx=src_vocab[0]['<pad>'], shuffle=False)

    # setup model
    model = NTRNN(input_size=len(src_vocab[0]), embedding_dim=args.embed_dim, hidden_size=args.hidden_dim, output_size=len(trg_vocab[0]), 
                  n_layers=4, dropout=args.dropout, device=device)

    # learning rate scheduler, optimizer, loss function
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if args.dropout <= 0.0:
        scheduler1 = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=5)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        LR_scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                                                   schedulers=[scheduler1, scheduler2],
                                                   milestones=[5])
    elif args.dropout > 0.0:
        scheduler1 = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=8)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        LR_scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                                                   schedulers=[scheduler1, scheduler2],
                                                   milestones=[8])

    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab[0]['<pad>'])

    # setup trainer
    trainer = Trainer(model, train_loader, test_loader, val_loader, optimizer, LR_scheduler, criterion, device)

    # training loop
    trainer.train(epochs)

    # test model
    trainer.model.load_state_dict(trainer.best_model_state)
    test_loss = trainer.evaluate()

    print("Final test loss: ", test_loss)

    # save model
    torch.save(trainer.model.state_dict(), save_path)