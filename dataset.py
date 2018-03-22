import numpy as np
from functools import partial
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset
import itertools
import json
import ipdb

LongTensor = torch.cuda.LongTensor if torch.cuda.device_count() and False else torch.LongTensor


def flatten(l):
    return list(itertools.chain.from_iterable(l))

class MLPLDataset(Dataset):
    def __init__(self, fname):
        with open(fname) as json_file:
            print('[MLPLDataset.init] Start data load')
            data = json.load(json_file)
            print('[MLPLDataset.init] End data load')
            inp, out = zip(*data)
            print('[MLPLDataset.init] split data to input/output')
            self.inp_vocab = ['<pad>'] + sorted(list(set(flatten(flatten(inp)))))
            print('[MLPLDataset.init] End create input vocab')
            self.out_vocab = sorted(list(set(out)))
            print('[MLPLDataset.init] End create output vocab')
            self.input_data = [[[self.inp_vocab.index(tok) for tok in seq] for seq in x] for x in inp]
            print('[MLPLDataset.init] End tokenize input data')
            self.output_data = [self.out_vocab.index(tok) for tok in out]
            print('[MLPLDataset.init] End tokenize output data')


    def __len__(self):
        return len(self.output_data)

    def __getitem__(self, idx):
        item = self.input_data[idx]
        lengths = [len(x) for x in item]
        max_length = max(lengths)
        rc = np.zeros((len(item),max_length))
        for i, it in enumerate(item):
            rc[i][:len(it)] = np.asarray(it)
        return rc, np.asarray(lengths), self.output_data[idx]
        return rc

def mlpl_collate(batch):
    inp, lengths, out = zip(*batch)
    x, y = zip(*[x.shape for x in inp])
    max_x = max(x)
    max_y = max(y)
    rc_inp = np.zeros((len(batch),max_x,max_y))
    rc_lengths = np.zeros((len(batch),max_x))
    for i,item in enumerate(inp):
        rc_inp[i,:x[i],:y[i]] = item
        rc_lengths[i,:x[i]] = lengths[i]
    return LongTensor(rc_inp), LongTensor(rc_lengths), LongTensor(out)
