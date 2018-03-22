import torch


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class MLPLSettings(metaclass=Singleton):
	def __init__(self, hyperparameters=None):
		self.hyperparameters = hyperparameters
		self.LongTensor = torch.cuda.LongTensor if hyperparameters['cuda'] else torch.LongTensor
		self.FloatTensor = torch.cuda.FloatTensor if hyperparameters['cuda'] else torch.FloatTensor
		self.ByteTensor = torch.cuda.ByteTensor if hyperparameters['cuda'] else torch.ByteTensor

	def __getitem__(self, key):
		return self.hyperparameters[key]

	def zeros(self,*size):
		rc = torch.zeros(*size)
		return rc.cuda() if self.hyperparameters['cuda'] else rc