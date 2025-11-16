import torch
import argparse
from model import NMTRNN
from dataset import load_data, get_data_loader, build_vocab, TextDataset
import torch.nn as nn
import torch.optim as optim
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

# Modified Trainer class with snapshot functionality
class TrainerWithSnapshots():
    def __init__(self, model, train_loader, test_loader, val_loader, optimizer, criterion, device, 
                 learning_rate_update_point=5, snapshot_dir="./model_snapshots/"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.learning_rate_update_point = learning_rate_update_point
        self.snapshot_dir = snapshot_dir

        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.loss_list = []
        self.global_step = 0
        
        # Create snapshot directory
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def save_model_snapshot(self, step, epoch, loss_value, snapshot_type="regular"):
        """Save model snapshot with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(
            self.snapshot_dir, 
            f"model_epoch_{epoch}_step_{step}_loss_{loss_value:.4f}_{snapshot_type}_{timestamp}.pt"
        )
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'epoch': epoch,
            'loss': loss_value,
            'timestamp': timestamp,
            'snapshot_type': snapshot_type,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }, snapshot_path)
        
        # Only print for critical snapshots to reduce I/O
        if snapshot_type in ['critical_step', 'best_model']:
            print(f"📸 Saved {snapshot_type} snapshot: {os.path.basename(snapshot_path)}")
        return snapshot_path

    def train_epoch(self, test_interval=None, snapshot_steps=None):
        """Optimized training epoch matching original trainer performance"""
        if snapshot_steps is None:
            snapshot_steps = []
            
        self.model.train()
        epoch_loss = 0
        iter = 0
        self.loss_list.append([])
        
        # Minimal tqdm exactly like original trainer
        for batch in tqdm(self.train_loader):
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(source, target[:-1], src_lengths=batch['src_lengths'], mode='train')

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = target[1:].view(-1)

            loss = self.criterion(output, trg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            iter += 1
            self.global_step += 1
            
            # Minimal snapshot checking (only when needed)
            if snapshot_steps and self.global_step in snapshot_steps:
                self.save_model_snapshot(
                    self.global_step, 
                    len(self.loss_list), 
                    loss.item(), 
                    snapshot_type="critical_step"
                )
            
            # Validation exactly like original trainer
            if test_interval and iter % test_interval == 0:
                self.loss_list[-1].extend([self.validate()])
                self.model.train()

        return epoch_loss / len(self.train_loader)
    
    def train(self, epochs, test_interval, snapshot_steps=None):
        """Modified training loop with snapshot tracking"""
        if snapshot_steps is None:
            # Default critical steps based on your loss spike observation
            snapshot_steps = [40, 45, 50, 55, 60, 65, 100, 150, 200]
        
        print(f"🎯 Will save snapshots at steps: {snapshot_steps}")
        
        for ep in range(epochs):
            print(f"\n🔄 Starting Epoch {ep+1}/{epochs}")
            
            # Save snapshot at epoch start
            if ep == 0:  # Save initial model
                self.save_model_snapshot(
                    self.global_step, ep+1, 0.0, snapshot_type="epoch_start"
                )
            
            train_loss = self.train_epoch(test_interval, snapshot_steps)
            val_loss = self.validate()
            
            print(f"Epoch: {ep+1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']}")
            
            # Learning rate scheduling
            if (ep+1) >= self.learning_rate_update_point:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.optimizer.param_groups[0]['lr'] *= 0.5
                new_lr = self.optimizer.param_groups[0]['lr']
                print(f"📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
                
                # Save snapshot when learning rate changes
                self.save_model_snapshot(
                    self.global_step, ep+1, val_loss, snapshot_type="lr_change"
                )
            
            # Save best model
            if val_loss <= self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                self.save_model_snapshot(
                    self.global_step, ep+1, val_loss, snapshot_type="best_model"
                )
    
    def evaluate(self):
        self.model.eval()
        loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(source, target[:-1], src_lengths=batch['src_lengths'], mode='test')
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)
                target = target[1:].view(-1)

                curr_loss = self.criterion(output, target)
                loss += curr_loss.item()
        return loss / len(self.test_loader)
    
    def validate(self):
        self.model.eval()
        loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(source, target[:-1], src_lengths=batch['src_lengths'], mode='test')
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)
                target = target[1:].view(-1)

                curr_loss = self.criterion(output, target)
                loss += curr_loss.item()
        return loss / len(self.val_loader)

# Argument parser (same as original)
parser = argparse.ArgumentParser()
parser.add_argument("--train_src", "-trs", default='./datasets/datasets_small/train.10k.en')
parser.add_argument("--test_src", "-tes", default='./datasets/datasets_small/test.100.en')
parser.add_argument("--train_trg", "-trt", default='./datasets/datasets_small/train.10k.de')
parser.add_argument("--test_trg", "-tet", default='./datasets/datasets_small/test.100.de')
parser.add_argument("--val_src", "-vls", default='./datasets/datasets_small/valid.100.en')
parser.add_argument("--val_trg", "-vlt", default='./datasets/datasets_small/valid.100.de')
parser.add_argument("--save_path", "-sp", default='./models/nmtrnn_snapshots.pt')
parser.add_argument("--vocab_path_base", "-vpb", default='./models/nmtrnn_snapshots_')
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
parser.add_argument("--window", "-win", type=int, default=10)
parser.add_argument("--attn_mode", "-attn", type=str, default='no')
parser.add_argument("--input_feeding", "-feed", type=bool, default=False)

# New snapshot-related arguments
parser.add_argument("--snapshot_steps", "-ss", nargs='+', type=int, 
                    default=[40, 45, 50, 55, 60], 
                    help="Steps at which to save model snapshots")
parser.add_argument("--snapshot_dir", "-sd", default="./model_snapshots/",
                    help="Directory to save model snapshots")

args = parser.parse_args()

filtered_sentence_file = {
    'train_en': None,
    'train_de': None,
    'test_en': None,
    'test_de': None,
    'dev_en': None,
    'dev_de': None,
}

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dict = vars(args)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("mps")

    print(f"🎯 Training with snapshots enabled")
    print(f"📁 Snapshots will be saved to: {args.snapshot_dir}")
    print(f"📸 Critical steps for snapshots: {args.snapshot_steps}")

    # Build vocab
    src_w2i, src_i2w = build_vocab(args.train_src, args.max_vocab)
    trg_w2i, trg_i2w = build_vocab(args.train_trg, args.max_vocab)

    # Load data
    src_train, trg_train = load_data(args.train_src, args.train_trg, src_w2i=src_w2i, trg_w2i=trg_w2i, max_len=args.max_len, 
                                   is_reverse=args.reverse, src_file=filtered_sentence_file['train_en'], 
                                   trg_file=filtered_sentence_file['train_de'], save=True)
    src_test, trg_test = load_data(args.test_src, args.test_trg, src_w2i=src_w2i, trg_w2i=trg_w2i, max_len=args.max_len, 
                                 is_reverse=args.reverse, src_file=filtered_sentence_file['test_en'], 
                                 trg_file=filtered_sentence_file['test_de'])
    src_val, trg_val = load_data(args.val_src, args.val_trg, src_w2i=src_w2i, trg_w2i=trg_w2i, max_len=args.max_len, 
                                is_reverse=args.reverse, src_file=filtered_sentence_file['dev_en'], 
                                trg_file=filtered_sentence_file['dev_de'])

    train_dataset = TextDataset(src_sentences=src_train, trg_sentences=trg_train, sos=trg_w2i['<sos>'], eos=trg_w2i['<eos>'])
    train_loader = get_data_loader(train_dataset, batch_size=args.batch_size, pad_idx=src_w2i['<pad>'], shuffle=True, drop_last=True)
    test_dataset = TextDataset(src_sentences=src_test, trg_sentences=trg_test, sos=trg_w2i['<sos>'], eos=trg_w2i['<eos>'])
    test_loader = get_data_loader(test_dataset, batch_size=args.test_batch_size, pad_idx=src_w2i['<pad>'], shuffle=False, drop_last=False)
    val_dataset = TextDataset(src_sentences=src_val, trg_sentences=trg_val, sos=trg_w2i['<sos>'], eos=trg_w2i['<eos>'])
    val_loader = get_data_loader(val_dataset, batch_size=args.test_batch_size, pad_idx=src_w2i['<pad>'], shuffle=False, drop_last=False)

    # Setup model
    model = NMTRNN(input_size=len(src_w2i), embedding_dim=args.embed_dim, hidden_size=args.hidden_dim, 
                  output_size=len(trg_w2i), n_layers=args.num_layer, dropout=args.dropout, 
                  input_feeding=args.input_feeding, attention_mode=args.attn_mode, attention_win=args.window, 
                  max_length=args.max_len, device=device, padding_idx=src_w2i['<pad>'])
    
    with torch.no_grad():
        model.encoder.embedding.weight[0] = torch.zeros(args.embed_dim)
        model.decoder.embedding.weight[0] = torch.zeros(args.embed_dim)
    model = model.to(device)

    # Setup optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_w2i['<pad>'])

    # Setup trainer with snapshot capability
    trainer = TrainerWithSnapshots(
        model, train_loader, test_loader, val_loader, optimizer, criterion, device, 
        learning_rate_update_point=args.learning_rate_update_point, 
        snapshot_dir=args.snapshot_dir
    )

    # Training loop - optimized for maximum performance
    test_interval = (train_dataset.__len__() // args.batch_size) // 20
    
    # Minimal snapshot steps - only save at critical moments
    if args.snapshot_steps:
        snapshot_steps = [step * test_interval for step in args.snapshot_steps[:3]]  # Limit to first 3 only
    else:
        snapshot_steps = []
    #print(snapshot_steps)
    start_time = datetime.now()
    
    trainer.train(args.epochs, test_interval, snapshot_steps)
    
    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training took: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

    # Test model
    trainer.model.load_state_dict(trainer.best_model_state)
    test_loss = trainer.evaluate()
    logs_dict["final_test_loss"] = test_loss
    print("Final test loss: ", test_loss)

    # Save final model
    torch.save(trainer.model.state_dict(), args.save_path)
    print(f"💾 Final model saved to: {args.save_path}")

    # Save vocab
    source_vocab_file = args.vocab_path_base + "src.pkl"
    target_vocab_file = args.vocab_path_base + "trg.pkl"

    src_vocab = {'w2i': src_w2i, 'i2w': src_i2w}
    trg_vocab = {'w2i': trg_w2i, 'i2w': trg_i2w}
    
    with open(source_vocab_file, "wb") as f:
        pickle.dump(src_vocab, f)
    with open(target_vocab_file, "wb") as f:
        pickle.dump(trg_vocab, f)

    # Save training log
    logs_dict["test_loss"] = trainer.loss_list
    logs_dict["src_vocab_file"] = source_vocab_file
    logs_dict["trg_vocab_file"] = target_vocab_file
    logs_dict["snapshot_dir"] = args.snapshot_dir
    logs_dict["total_snapshots_saved"] = len([f for f in os.listdir(args.snapshot_dir) if f.endswith('.pt')])
    
    log_file = f'./logs/training_log_snapshots_{timestamp}.json'
    with open(log_file, 'w') as f:
        json.dump(logs_dict, f, indent=2)

    print(f"📊 Training log saved to: {log_file}")
    print(f"📸 Total snapshots saved: {logs_dict['total_snapshots_saved']}")
    print("🎉 Training with snapshots complete!")