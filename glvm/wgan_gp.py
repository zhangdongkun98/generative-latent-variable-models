import rllib
from rllib.template.model import FeatureMapper

import torch
import torch.nn as nn
from torch.optim import RMSprop


class WGANGP(rllib.template.Method):
    lr_g = 1e-4
    lr_d = 1e-4
    weight_decay = 5e-4

    def __init__(self, config: rllib.basic.YamlConfig, writer: rllib.basic.Writer):
        '''
        '''

        super().__init__(config, writer)

        self.generator = config.get('net_generator', Generator)(config, ).to(self.device)
        self.discriminator = config.get('net_discriminator', Discriminator)(config, ).to(self.device)
        self.models_to_save = [self.generator, self.discriminator]

        self.g_optimizer = RMSprop(self.generator.parameters(), lr=self.lr_g, weight_decay=self.weight_decay)
        self.d_optimizer = RMSprop(self.discriminator.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)

        self.train_dataloader = None


    def update_parameters(self):
        super().update_parameters()

        data = self.train_dataloader.do_sth()

        '''discriminator'''
        rllib.basic.torch.set_requires_grad(self.discriminator, True)
        self.d_optimizer.zero_grad()

        ###

        self.d_optimizer.step()

        '''generator'''
        rllib.basic.torch.set_requires_grad(self.discriminator, False)
        self.g_optimizer.zero_grad()

        ###


        self.g_optimizer.step()







class Generator(rllib.template.Model):
    def __init__(self, config, dim_input, dim_output):
        super(Generator, self).__init__(config, model_id=0)

        self.fe = FeatureMapper(config, 0, dim_input, dim_output)
        self.apply(rllib.utils.init_weights)
    
    def forward(self, x):
        return self.fe(x)


class Discriminator(rllib.template.Model):
    def __init__(self, config, dim_input, dim_output):
        super(Discriminator, self).__init__(config, model_id=0)

        self.fe = FeatureMapper(config, 0, dim_input, dim_output)
        self.apply(rllib.utils.init_weights)

    def forward(self, x):
        return self.fe(x)



