import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value
from collections import Counter
import ipdb
from model import *
from dataset import *
from settings import *
from MLPLVocab import *

INIT_LR = 0.01
BATCH_SIZE = 16

TRAIN = True

NUM_EPOCHS = 400
PRINT_LOSS_EVERY = 10
PRINT_ACC_EVERY = 2
LOG_EVERY = 10
SAVE_EVERY = 5

max_size = 500
min_freq = 1

criterion = nn.CrossEntropyLoss()

settings = MLPLSettings({'cuda': False, 'base_model': 'model_epoch_10.model'})

def train(ds, model, optimizer=None, iters=None, ds_validate=None, do_log=True):
  print('Start training')
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

def eval_(ds, model):
    sampler = torch.utils.data.sampler.SequentialSampler(ds)
    testloader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,
                                                sampler = sampler,
                                                collate_fn = mlpl_collate)
    get_step = lambda x,y: x*len(testloader)+y
    current_time = time.time()
    accuracy = 0.
    total_correct = 0.
    for i, (seq, seq_len, labels) in enumerate(testloader):
        seq = Variable(seq)
        labels = Variable(labels)
        if settings['cuda']:
            seq, seq_len, labels = seq.cuda(), seq_len.cuda(), labels.cuda()
        outputs = model(seq,seq_len)
        preds = outputs.max(1)[1]
        correct = preds == labels
        total_correct += int(torch.sum(correct).data.numpy())
    # ipdb.set_trace()
    print('Accuracy is {}'.format(total_correct/len(ds)))

        # running_accuracy +=
        # if i % PRINT_ACC_EVERY == PRINT_ACC_EVERY-1:
        #     new_time = time.time()
        #     print('Average time per mini-batch, %f' % ((new_time-current_time) / PRINT_LOSS_EVERY))
        #     current_time = new_time
        #     print('[%d, %5d] accuracy: %.3f' %
        #         (epoch + 1, i + 1, running_accuracy / PRINT_LOSS_EVERY))


def main():
    ds = None
    with open('train_freq_10.json') as freq_file:
        freqs = [Counter(x) for x in json.load(freq_file)]

    v_in = MLPLVocab(freqs[0], max_size=max_size, min_freq=min_freq, specials=['<pad>', '<unk>'])
    v_out = MLPLVocab(freqs[1], max_size=max_size, min_freq=min_freq)

    ftrain = 'train_data_10.json'
    feval = 'eval_data_10.json'

    model = MLPLEncoder(len(v_in), len(v_out), 32)
    if TRAIN:
        ds  = MLPLDataset(ftrain, v_in, v_out)
    else:
        ds  = MLPLDataset(feval, v_in, v_out)
        model.load_state_dict(torch.load('{}'.format(settings['base_model'])))

    if settings['cuda']:
        model = model.cuda()
    if TRAIN:
        train(ds,model)
    else:
        eval_(ds, model)


if __name__=='__main__':
    main()
