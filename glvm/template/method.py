import rllib

import time

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Method(rllib.template.Method):
    """
        Method with dataset.
    """

    evaluate_interval = 2

    def __init__(self, config: rllib.basic.YamlConfig, writer: rllib.basic.Writer):
        super().__init__(config, writer)

        self.step_epoch = -1
        self.step_evaluate = -1

        self.train_dataset: Dataset = None
        self.evaluate_dataset: Dataset = None
        self.train_dataloader: DataLoader = None
        self.evaluate_dataloader: DataLoader = None
        return


    def update_parameters_(self):
        self.step_epoch += 1
        print('\n\nepoch index: ', self.step_epoch)
        t1 = time.time()
        for i, data_samples in enumerate(self.train_dataloader):
            print('    train ratio: {} / {}'.format(i, len(self.train_dataset)/self.batch_size), end='\r', flush=True)
            self.update_parameters(rllib.basic.Data(**data_samples).to(self.device))
        t2 = time.time()
        print('\ntrain one epoch time: ', t2-t1, 's')

        if self.step_epoch % self.evaluate_interval == 0:
            for i, data_samples in enumerate(self.evaluate_dataloader):
                print('    evaluate ratio: {} / {}'.format(i, len(self.evaluate_dataset)/self.batch_size), end='\r', flush=True)
                with torch.no_grad():
                    self.evaluate_parameters(rllib.basic.Data(**data_samples).to(self.device))
            t3 = time.time()
            print('\nevaluate one epoch time: ', t3-t2, 's')
        return



    def evaluate_parameters(self, data):
        return

