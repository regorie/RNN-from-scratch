import torch
import torch.utils.data as data
import torch.nn as nn
from collections import Counter
from tqdm import tqdm
import time
import pickle

def build_vocab(file_name, max_vocab):
    print("Building vocab...")
    w2i = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}
    i2w = {0:'<pad>', 1:'<unk>', 2:'<sos>', 3:'<eos>'}

    counter = Counter()

    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for line in tqdm(lines):
            for token in line.strip().split(' '):
                counter[token] += 1

        id = 4
        for token, _ in counter.most_common(max_vocab):
            w2i[token] = id
            i2w[id] = token
            id += 1
    return w2i, i2w

def load_data(src_data_path, trg_data_path, src_w2i, trg_w2i, max_len=100, is_reverse=False, src_file=None, trg_file=None, save=False):

    print("Loading data...")

    if src_file is not None and trg_file is not None:
        with open(src_file, 'rb') as sf, open(trg_file, 'rb') as tf:

            src_sentences = pickle.load(sf)
            trg_sentences = pickle.load(tf)

            if is_reverse:
                src_sentences = src_sentences[::-1]
                trg_sentences = trg_sentences[::-1]
            
            return src_sentences, trg_sentences

    with open(src_data_path, 'r', encoding='utf-8') as sf,\
         open(trg_data_path, 'r', encoding='utf-8') as tf:
        src_lines = sf.readlines()
        trg_lines = tf.readlines()

        src_sentences_all = []
        trg_sentences_all = []

        remove_idx = []

        for idx, (src_line, trg_line) in enumerate(zip(src_lines, trg_lines)):
            if len(src_line.strip().split(' ')) > max_len or len(trg_line.strip().split(' ')) > max_len:
                remove_idx.append(idx)

            src_sentences_all.append(src_line.strip().split(' '))
            trg_sentences_all.append(trg_line.strip().split(' '))

        start_time = time.time()
        src_sentences = [sen for idx, sen in tqdm(enumerate(src_sentences_all), desc="Filtering source sentences", total=len(src_sentences_all)) if idx not in remove_idx]
        trg_sentences = [sen for idx, sen in tqdm(enumerate(trg_sentences_all), desc="Filtering target sentences", total=len(trg_sentences_all)) if idx not in remove_idx]
        
        # save filtered sentences

        end_time = time.time()
        print("took ", end_time-start_time, " to filter sentences")
        del src_sentences_all
        del trg_sentences_all
        del remove_idx

        print("Mapping source sentence...")
        for sentence in tqdm(src_sentences):
            for i, word in enumerate(sentence):
                if word in src_w2i:
                    sentence[i] = src_w2i[word]
                else:
                    sentence[i] = src_w2i['<unk>']
        print("Mapping target sentence...")
        for sentence in tqdm(trg_sentences):
            for i, word in enumerate(sentence):
                if word in trg_w2i:
                    sentence[i] = trg_w2i[word]
                else:
                    sentence[i] = trg_w2i['<unk>']
        
        if save:
            with open('src_sentences_filtered_train.pkl', 'wb') as f:
                pickle.dump(src_sentences, f)
            with open('trg_sentences_filtered_train.pkl', 'wb') as f:
                pickle.dump(trg_sentences, f)

        if is_reverse:
            src_sentences = src_sentences[::-1]
            trg_sentences = trg_sentences[::-1]

    return src_sentences, trg_sentences

def get_collate_fn(pad_idx):
    def collate_fn(batch):
        """
        batch: list of dicts with keys 'source', 'decoder_input', 'decoder_output'
        """
        sources = [item['source'] for item in batch]
        targets = [item['target'] for item in batch]

        src_lengths = [len(src) for src in sources]

        # Pad sequences to the max length in the batch
        src_padded = nn.utils.rnn.pad_sequence(sources, padding_value=pad_idx, batch_first=False)
        target_padded = nn.utils.rnn.pad_sequence(targets, padding_value=pad_idx, batch_first=False)

        batch = {
            'source': src_padded,
            'target': target_padded,
            'src_lengths': torch.tensor(src_lengths)
        }
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_idx, shuffle=False, drop_last=False):
    collate_fn = get_collate_fn(pad_idx)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last  # Ensure all batches are of equal size
    )
    return data_loader

class TextDataset(data.Dataset):
    def __init__(self, src_sentences, trg_sentences, sos, eos):
        """
        src_sentences: list of sentences(list of word idices)
        trg_sentences: same as above
        src_vocab = (src_w2i, src_i2w)
        trg_vocab = same as above
        """
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        for i in range(len(trg_sentences)):
            self.trg_sentences[i] = [sos] + self.trg_sentences[i] + [eos]
        #self.src_vocab = src_vocab
        #self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_seq = self.src_sentences[idx]
        trg_seq = self.trg_sentences[idx]

        # 1. source: input to encoder
        # 2. target: target sequence
        #            use target[:-1] for decoder input
        #            use target[1:] for decoder output

        return { 
            'source': torch.tensor(src_seq, dtype=torch.long),
            'target': torch.tensor(trg_seq, dtype=torch.long)
        }