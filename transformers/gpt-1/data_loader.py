import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torch.functional as F
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from collections import Counter
import string
data="./data/input.txt"
with open(data,'r') as f:
    text=f.read()
    

def doc2words(doc):
    lines = doc.split('\n')
    lines = [line.strip(r'\"') for line in lines]
    words = ' '.join(lines).split()

    punct = set(string.punctuation)
    words = [''.join([char for char in list(word) if char not in punct]) for word in words]
    return words



words=doc2words(text)
def getvocab(words):
    wordfreq = Counter(words)
    sorted_wordfreq = sorted(wordfreq, key=wordfreq.get)
    return sorted_wordfreq

vocab=getvocab(words)

def vocab_map(vocab):
    int_to_vocab = {k:w for k,w in enumerate(vocab)}
    vocab_to_int = {w:k for k,w in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

idx2word,word2idx=vocab_map(vocab)


class TextDataset(Dataset):
    def __init__(self, words, vocab_to_int, seq_size):
        # Convert words to integers
        word_ints = [vocab_to_int[word] for word in words]
        num_batches = len(word_ints) // seq_size
        Xs = word_ints[:num_batches * seq_size]  # Trim to fit seq_size
        self.Xs = np.reshape(Xs, (-1, seq_size))  # Shape: (num_batches, seq_size)

    def __len__(self):
        return len(self.Xs)  # Number of batches

    def __getitem__(self, idx):
        return torch.tensor(self.Xs[idx], dtype=torch.long)


seq_size = 32
batch_size = 64


dataset = TextDataset(words, word2idx, seq_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
