import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from collections import defaultdict


def load_vocab(text):

    word_to_idx = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
    idx_to_word = {0:'<pad>', 1:'<sos>', 2:'<eos>', 3:'<unk>'}

    idx = 4
    for sentence in text:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx_to_word[idx] = word
                idx += 1

    return word_to_idx, idx_to_word

def load_data(src_data_path, trg_data_path, max_len=100, is_reverse=False, vocab=None):
    # vocab : (src_w2i, trg_w2i)
    with open(src_data_path, 'r', encoding='utf-8') as sf,\
         open(trg_data_path, 'r', encoding='utf-8') as tf:
        src_lines = sf.readlines()
        trg_lines = tf.readlines()

        src_sentences = []
        trg_sentences = []

        for src_line, trg_line in zip(src_lines, trg_lines):
            src_sentences.append(src_line.strip().split(' ')[:max_len])
            trg_sentences.append(trg_line.strip().split(' ')[:max_len])

        src_w2i, src_i2w, trg_w2i, trg_i2w = None, None, None, None
        if vocab is None:
            src_w2i, src_i2w = load_vocab(src_sentences)
            trg_w2i, trg_i2w = load_vocab(trg_sentences)

            for sentence in src_sentences:
                for i, word in enumerate(sentence):
                    sentence[i] = src_w2i[word]
            for sentence in trg_sentences:
                for i, word in enumerate(sentence):
                    sentence[i] = trg_w2i[word]
        else:
            for sentence in src_sentences:
                for i, word in enumerate(sentence):
                    if word in vocab[0]:
                        sentence[i] = vocab[0][word]
                    else:
                        sentence[i] = vocab[0]['<unk>']
            for sentence in trg_sentences:
                for i, word in enumerate(sentence):
                    if word in vocab[1]:
                        sentence[i] = vocab[1][word]
                    else:
                        sentence[i] = vocab[1]['<unk>']

        if is_reverse:
            sentence = reversed(sentence)
            trg_sentences = reversed(trg_sentences)

    return src_sentences, trg_sentences, (src_w2i, src_i2w), (trg_w2i, trg_i2w)

def get_collate_fn(pad_idx):
    def collate_fn(batch):
        """
        batch: list of dicts with keys 'source', 'decoder_input', 'decoder_output'
        """
        sources = [item['source'] for item in batch]
        targets = [item['target'] for item in batch]

        # Pad sequences to the max length in the batch
        src_padded = nn.utils.rnn.pad_sequence(sources, padding_value=pad_idx, batch_first=False)
        target_padded = nn.utils.rnn.pad_sequence(targets, padding_value=pad_idx, batch_first=False)

        batch = {
            'source': src_padded,
            'target': target_padded
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
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab):
        """
        src_sentences: list of sentences(list of word idices)
        trg_sentences: same as above
        src_vocab = (src_w2i, src_i2w)
        trg_vocab = same as above
        """
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_seq = self.src_sentences[idx]
        trg_seq = self.trg_sentences[idx]
        trg_seq = [self.trg_vocab[0]['<sos>']] + trg_seq + [self.trg_vocab[0]['<eos>']]

        # 1. source: input to encoder
        # 2. target: target sequence
        #            use target[:-1] for decoder input
        #            use target[1:] for decoder output

        return { 
            'source': torch.tensor(src_seq, dtype=torch.long),
            'target': torch.tensor(trg_seq, dtype=torch.long)
        }