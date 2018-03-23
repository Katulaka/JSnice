import json
import itertools
from collections import Counter
from MLPLVocab import *

def flatten(l):
    return list(itertools.chain.from_iterable(l))

MAX_CONTEXT = 100

def clean_data(path_r, path_w, count_f):
    c_inp = Counter()
    c_out = Counter()
    with open(path_r) as js_file:
        data = js_file.readlines()
        clean_data = []
        for d in data:
            ds = json.loads(d) #convert from string to dict
            if len(ds['input']) <= MAX_CONTEXT:
                inp = [[el for el in dd if el!= '0PAD'] for dd in ds['input']]
                if count_f is not None:
                    c_inp.update(flatten(inp))
                    c_out.update([ds['output']])
                clean_data.append((inp, ds['output']))

    with open(path_w, 'w') as outfile:
        json.dump(clean_data, outfile)
    if count_f:
        with open(count_f, 'w') as outfile:
            json.dump([c_inp, c_out], outfile)
    # return clean_data, [c_inp, c_out]


train_in = 'train_100.txt'
train_out = 'train_data_{}'.format(MAX_CONTEXT)
train_freq = 'train_freq_{}'.format(MAX_CONTEXT)
eval_in = 'eval_10.txt'
eval_out = 'eval_data_{}'.format(MAX_CONTEXT)

clean_data(train_in, train_out, train_freq)
clean_data(eval_in, eval_out, None)
