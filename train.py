import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

from model import *
from dataset import *
from settings import *

INIT_LR = 0.01
BATCH_SIZE = 128

NUM_EPOCHS = 400
PRINT_LOSS_EVERY = 2
LOG_EVERY = 10
SAVE_EVERY = 5

criterion = nn.CrossEntropyLoss()

settings = MLPLSettings({'cuda': True})

def train(ds, model, optimizer=None, iters=None, ds_validate=None, do_log=True):

  if optimizer is None:
    optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9)

  sampler = torch.utils.data.sampler.RandomSampler(ds)
  trainloader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, sampler = sampler, collate_fn = mlpl_collate)

  if do_log:
    configure("runs/mlpl_run_%s" % str(time.time())[-4:], flush_secs=5)
  get_step = lambda x,y: x*len(trainloader)+y
  start_epoch=0
  current_time = time.time()

  for epoch in range(start_epoch,NUM_EPOCHS):
    import ipdb; ipdb.set_trace()
    running_loss = 0.
    total_correct = 0.
    # utils.exp_lr_scheduler(optimizer, epoch, init_lr=settings['init_lr'], lr_decay_epoch=settings['decay_num_epochs'],decay_rate=settings['decay_lr'])
    for i, (seq, seq_len, labels) in enumerate(trainloader):
      seq = Variable(seq)
      labels = Variable(labels)
      if settings['cuda']:
        seq, seq_len, labels = seq.cuda(), seq_len.cuda(), labels.cuda()

      optimizer.zero_grad()
      outputs = model(seq,seq_len)
      loss = criterion(outputs, labels)   # + torch.sum(aux_losses)
      try:
        loss.backward()
      except RuntimeError as e:
        print('Woah, something is going on')
        print(e)
      optimizer.step()
      running_loss += loss.data[0]
      if get_step(epoch,i) % LOG_EVERY == 0:
        log_value('loss',loss.data[0],get_step(epoch,i))
      if i % PRINT_LOSS_EVERY == PRINT_LOSS_EVERY-1:
        new_time = time.time()
        print('Average time per mini-batch, %f' % ((new_time-current_time) / PRINT_LOSS_EVERY))
        current_time = new_time
        # ipdb.set_trace()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / PRINT_LOSS_EVERY))
        running_loss = 0.0
        # print('[%d, %5d] training accuracy: %.3f' %
        #       (epoch + 1, i + 1, total_correct / (PRINT_LOSS_EVERY*BATCH_SIZE)))
      if iters is not None and get_step(epoch,i) > iters:
        return

    if epoch % SAVE_EVERY == 0:
      torch.save(model.state_dict(),'model_epoch_%d.model' % epoch)



def main():
    fname = 'train_data_100.json'
    import ipdb; ipdb.set_trace()
    ds = MLPLDataset(fname)
    model = MLPLEncoder(len(ds.inp_vocab), len(ds.out_vocab), 32)
    if settings['cuda']:
        model = model.cuda()
    train(ds,model)


if __name__=='__main__':
    main()
