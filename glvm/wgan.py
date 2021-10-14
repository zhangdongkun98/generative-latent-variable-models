import rllib
from rllib.template.model import FeatureMapper

import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



def clamp_module(net: nn.Module, w_min, w_max):
    for module in net.children():
        clamp_module(module, w_min, w_max)

    # if net.__class__.__name__ == 'Linear':
    if isinstance(net, nn.Linear):
        # print('here')
        net.weight.requires_grad = False
        net.weight.clamp_(w_min, w_max)
        net.weight.requires_grad = True
    return



class WGAN(rllib.template.Method):
    lr_g = 1e-4
    lr_d = 1e-4
    weight_decay = 5e-4
    batch_size = 32

    dim_noise = 64

    num_workers = 16

    save_model_interval = 200

    def __init__(self, config: rllib.basic.YamlConfig, writer: rllib.basic.Writer):
        '''
        '''

        super().__init__(config, writer)

        config.set('dim_noise', self.dim_noise)

        self.discriminator = config.get('net_discriminator', Discriminator)(config).to(self.device)
        self.generator = config.get('net_generator', Generator)(config).to(self.device)
        self.models_to_save = [self.discriminator, self.generator]

        self.d_optimizer = RMSprop(self.discriminator.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
        self.g_optimizer = RMSprop(self.generator.parameters(), lr=self.lr_g, weight_decay=self.weight_decay)

        dataset_cls = config.get('dataset_cls', Dataset)
        # train_dataloader = DataLoader(dataset_cls(config, mode='train'), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        train_dataloader = DataLoader(dataset_cls(config, mode='train'), batch_size=self.batch_size, shuffle=False, num_workers=1)
        # evaluate_dataloader = DataLoader(dataset_cls(config, mode='evaluate'), batch_size=1, shuffle=False, num_workers=1)
        evaluate_dataloader = DataLoader(dataset_cls(config, mode='evaluate'), batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.train_samples, self.evaluate_samples = iter(train_dataloader), iter(evaluate_dataloader)
        return



    def update_parameters(self):
        super().update_parameters()

        data = rllib.basic.Data(**next(self.train_samples)).to(self.device)

        '''discriminator'''
        rllib.basic.pytorch.set_requires_grad(self.discriminator, True)
        self.d_optimizer.zero_grad()
        d_loss, info = self.calculate_d_loss(data)
        d_loss.backward()
        torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), clip_value=1)
        self.d_optimizer.step()

        ### weight Clipping WGAN
        c = 0.005
        c = 0.01
        clamp_module(self.discriminator, -c, c)




        '''generator'''
        rllib.basic.pytorch.set_requires_grad(self.discriminator, False)
        self.g_optimizer.zero_grad()
        g_loss = self.calculate_g_loss(data, info)
        g_loss.backward()
        torch.nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=1)
        self.g_optimizer.step()

        self.writer.add_scalar('loss/d_loss', d_loss.detach().item(), self.step_update)
        self.writer.add_scalar('loss/g_loss', g_loss.detach().item(), self.step_update)

        if self.step_update % self.save_model_interval == 0:
            self._save_model()
        
        return


    def calculate_d_loss(self, data):
        return torch.tensor(0.0)
    
    def calculate_g_loss(self, data):
        return torch.tensor(0.0)




class Generator(rllib.template.Model):
    def __init__(self, config):
        super(Generator, self).__init__(config, model_id=0)

        ###! warning   todo
        dim_input, dim_output = 1, 1

        self.fe = FeatureMapper(config, 0, dim_input, dim_output)
        self.apply(rllib.utils.init_weights)
    
    def forward(self, x):
        return self.fe(x)


class Discriminator(rllib.template.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__(config, model_id=0)

        ###! warning   todo
        dim_input, dim_output = 1, 1

        self.fe = FeatureMapper(config, 0, dim_input, dim_output)
        self.apply(rllib.utils.init_weights)

    def forward(self, x):
        return self.fe(x)



