from torchtext import vocab

class MLPLVocab(vocab.Vocab):
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'],
                 vectors=None, unk_init=None, vectors_cache=None):
      super(MLPLVocab, self).__init__(counter, max_size=max_size, min_freq=min_freq,
                                        specials=['<pad>', '<unk>', '<var>'])

    def vtoi(self, val):
        if val.startswith('MLPL'):
            return (self.stoi.get('<var>'),int(val.split('_')[1]))
        id = self.stoi.get(val)
        if id:
            return (id,0)
        return (self.stoi.get('<unk>'),0)

    def itov(self, id):
        try:
            return self.itos[id]
        except:
            print('number is not defined')
            raise
