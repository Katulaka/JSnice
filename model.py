import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext
import ipdb

from settings import *

class MLPLEncoder(nn.Module):
    def __init__(self, inp_vocab_size, out_vocab_size, hidden_size,
        input_dropout_p=0, dropout_p=0,
        n_layers=1, bidirectional=False, rnn_cell='gru'):
      super(MLPLEncoder, self).__init__()


      self.settings = MLPLSettings()
      if rnn_cell.lower() == 'lstm':
        self.rnn_cell = nn.LSTM
      elif rnn_cell.lower() == 'gru':
        self.rnn_cell = nn.GRU
      else:
        raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

      self.embedding = nn.Embedding(inp_vocab_size, hidden_size)
      self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
                               batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
      self.output_layer = nn.Linear(hidden_size,out_vocab_size)
      self.input_dropout_p = input_dropout_p
      self.input_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, input_var, input_lengths):
      """
      Applies a multi-layer RNN to an input sequence.

      Args:
          input_var (batch, seq_len): tensor containing the features of the input sequence.
          input_lengths (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch

      Returns: output, hidden
          - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
          - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
      """
      inp = input_var.view(-1,input_var.size(2))
      seq_lengths, perm_idx = input_lengths.view(-1).sort(0, descending=True)
      seq_tensor = inp[perm_idx]
      np_lengths = seq_lengths.cpu().numpy()
      if 0 in seq_lengths:
        max_non_zero = seq_lengths[seq_lengths>0].size(0)
        compact_seq_tensor = seq_tensor[:max_non_zero]
        compact_seq_lengths = np_lengths[:max_non_zero]
      else:
        compact_seq_tensor = seq_tensor
        compact_seq_lengths = np_lengths

      embedded = self.embedding(compact_seq_tensor)
      embedded = self.input_dropout(embedded)
      embedded = nn.utils.rnn.pack_padded_sequence(embedded, compact_seq_lengths, batch_first=True)
      output, hidden = self.rnn(embedded)
      hidden = hidden.squeeze(0)
      if 0 in seq_lengths:
        padded_hidden = torch.cat([hidden,Variable(self.settings.zeros(inp.size(0)-max_non_zero,hidden.size(1)))])
      else:
        padded_hidden = hidden
      orig_hidden = torch.zeros_like(padded_hidden)
      orig_hidden[perm_idx]=padded_hidden
      orig_hidden = orig_hidden.view(input_var.size(0),input_var.size(1),-1)
      context_counts = Variable(torch.sum(input_lengths>0,1).float()).unsqueeze(1)      
      record_hidden = torch.sum(orig_hidden,1)
      record_hidden = record_hidden / context_counts.expand_as(record_hidden)
      # output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
      logits = self.output_layer(record_hidden)
      return logits



      return output, hidden
