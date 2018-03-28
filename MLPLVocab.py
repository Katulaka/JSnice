from torchtext import vocab

class MLPLVocab(vocab.Vocab):
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'],
                 vectors=None, unk_init=None, vectors_cache=None):
      super(MLPLVocab, self).__init__(counter, max_size=max_size, min_freq=min_freq,
                                        specials=['<pad>', '<unk>'])

    def vtoi(self, val):
        id = self.stoi.get(val)
        if id:
            return id
        return self.stoi.get('<unk>')

    def itov(self, id):
        try:
            return self.itos[id]
        except:
            print('number is not defined')
            raise
