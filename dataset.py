import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import ipdb

LongTensor = torch.cuda.LongTensor if torch.cuda.device_count() and False else torch.LongTensor

class MLPLDataset(Dataset):
    def __init__(self, fname, inp_vocab, out_vocab):
        print('[MLPLDataset.init] Start init')


        self.inp_vocab = inp_vocab
        self.out_vocab = out_vocab
        self.input_data = []
        self.output_data = []

        with open(fname) as json_file:

            data = json.load(json_file)
            print('[MLPLDataset.init] End data load')
        inp, out = zip(*data)
        print('[MLPLDataset.init] split data to input/output')

        def serialize_number(d, idx):
            idx_mjr, idx_mnr = idx
            if idx_mnr == 0:
                return idx
            if not idx_mnr in d.keys():
                d[idx_mnr] = len(d)+1
            return idx_mjr, d[idx_mnr]

        for rec,y in zip(inp,out):
            if self.out_vocab.vtoi(y)[0]==self.out_vocab.vtoi('<unk>')[0]:
                continue

            d = {}
            self.output_data.append(self.out_vocab.vtoi(y)[0])
            self.input_data.append([[serialize_number(d,self.inp_vocab.vtoi(tok)) for tok in ctx] for ctx in rec])

            # new_data = []            
            # new_mask = []            
            # for ctx in x:
            #     s1, s2 = zip(*[self.inp_vocab.vtoi(tok) for tok in ctx])
            #     new_data.append(s1)
            #     new_mask.append(s2)

        # new_inp, new_out = zip(*[(x,y) for x,y in zip(inp,out) if self.out_vocab.vtoi(y)[0]!=self.out_vocab.vtoi('<unk>')[0]])
        # print('[MLPLDataset.init] removed unk labels')

        # self.input_data =  [[[self.inp_vocab.vtoi(tok) for tok in seq]
        #                         for seq in x] for x in new_inp]
        # print('[MLPLDataset.init] finished input_data')

        # self.output_data = [self.out_vocab.vtoi(tok) for tok in new_out]
        # print('[MLPLDataset.init] finished output_data')

    def __len__(self):
        return len(self.output_data)

    def __getitem__(self, idx):
        item = self.input_data[idx]
        lengths = [len(x) for x in item]
        max_length = max(lengths)
        rc = np.zeros((len(item),max_length,2))
        for i, it in enumerate(item):
            rc[i][:len(it)] = np.asarray(it)
        return rc, np.asarray(lengths), self.output_data[idx]
        return rc

def mlpl_collate(batch):
    inp, lengths, out = zip(*batch)
    x, y, _ = zip(*[x.shape for x in inp])
    max_x = max(x)
    max_y = max(y)
    rc_inp = np.zeros((len(batch),max_x,max_y,2))
    rc_lengths = np.zeros((len(batch),max_x))
    for i,item in enumerate(inp):
        rc_inp[i,:x[i],:y[i]] = item
        rc_lengths[i,:x[i]] = lengths[i]
    return LongTensor(rc_inp), LongTensor(rc_lengths), LongTensor(out)
