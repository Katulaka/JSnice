import json

MAX_CONTEXT = 100
# for path_r, path_w in zip(f_in, f_out):
def clean_data(path_r, path_w):
    with open(path_r) as js_file:
        data = js_file.readlines()
        clean_data = []
        for d in data:
            ds = json.loads(d)
            if len(ds['input']) <= MAX_CONTEXT:
                inp = [[el for el in dd if el!= '0PAD'] for dd in ds['input']]
                clean_data.append((inp, ds['output']))

    with open(path_w, 'w') as outfile:
        json.dump(clean_data, outfile)
    return clean_data

def gen_vocab(inp, out):
    import ipdb; ipdb.set_trace()
    inp_vocab = ['<pad>'] + sorted(list(set(flatten(flatten(inp)))))
    inp_vocab = dict(map(reversed, enumerate(self.inp_vocab)))
    inp_vocab.update({'<unk>':-1})
    print('[gen_vocab] End create input vocab')
    out_vocab = sorted(list(set(out)))
    out_vocab = dict(map(reversed, enumerate(self.out_vocab)))
    out_vocab.update({'<unk>':-1})
    print('[gen_vocab] End create output vocab')
    with open('vocab', 'w') as outfile:
        json.dump([inp_vocab, out_vocab], outfile)

def main():

    in_train = 'mljs/jsnice_data/training_processed.txt'
    out_train =  'train_data_{}.json'.format(MAX_CONTEXT)
    tdata = clean_data(in_train, out_train)
    gen_vocab(*zip(*tdata))

    in_eval =  'mljs/jsnice_data/eval_processsed.txt'
    out_eval =  'eval_data_{}.json'.format(MAX_CONTEXT)
    _ = clean_data(f_eval, out_eval)

if __name__=='__main__':
    main()
