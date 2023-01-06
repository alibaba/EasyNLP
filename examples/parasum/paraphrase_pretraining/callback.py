import os
import torch
import sys
from torch import nn
from os.path import join

from fastNLP.core.callback import Callback

class MyCallback(Callback):
    def __init__(self, args):
        super(MyCallback, self).__init__()
        self.args = args
        self.real_step = 0
    
    def on_valid_begin(self):
        with open(join(self._trainer.save_path, 'train_info.txt'), 'a') as f:
            print('Current step is: {}'.format(self.step), file=f)

    def on_step_end(self):
        # warm up
        if self.step % self.update_every == 0 and self.step > 0:
            self.real_step += 1
            cur_lr = self.args.max_lr * 100 * min(self.real_step ** (-0.5), self.real_step * self.args.warmup_steps**(-1.5))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

            if self.real_step % 1000 == 0:
                self.pbar.write('Current learning rate is {:.8f}, real_step: {}'.format(cur_lr, self.real_step))
    
    def on_epoch_end(self):
        self.pbar.write('Epoch {} is done !!!'.format(self.epoch))
    
