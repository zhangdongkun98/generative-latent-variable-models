import rllib
from rllib.template.model import FeatureMapper
import glvm

import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.autograd import grad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class WganGp(glvm.template.Method):
    lr_g = 1e-4
    lr_d = 1e-4
    weight_decay = 5e-4
    batch_size = 32

    dim_noise = 64

    num_workers = 16

    evaluate_interval = 2
    save_model_interval = 2000

    def __init__(self, config: rllib.basic.YamlConfig, writer: rllib.basic.Writer):
        '''
        '''

        super().__init__(config, writer)

        config.set('dim_noise', self.dim_noise)

        self.discriminator = config.get('net_discriminator', Discriminator)(config).to(self.device)
        self.generator = config.get('net_generator', Generator)(config).to(self.device)
        self.models_to_save = [self.discriminator, self.generator]
        self.models_to_load = [self.discriminator, self.generator]

        self.d_optimizer = RMSprop(self.discriminator.parameters(), lr=self.lr_d, weight_decay=self.weight_decay)
        self.g_optimizer = RMSprop(self.generator.parameters(), lr=self.lr_g, weight_decay=self.weight_decay)

        dataset_cls = config.get('dataset_cls', Dataset)
        self.train_dataset = dataset_cls(config, mode='train')
        self.evaluate_dataset = dataset_cls(config, mode='evaluate')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True)
        self.evaluate_dataloader = DataLoader(self.evaluate_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True)
        return


    def update_parameters(self, data):
        self.update_parameters_start()

        # noise, fake, info = self.forward(data)

        '''discriminator'''
        rllib.basic.pytorch.set_requires_grad(self.discriminator, True)
        self.d_optimizer.zero_grad()
        # d_loss, info = self.calculate_d_loss(data, noise, fake, info)
        d_loss, info = self.calculate_d_loss(data)   ### v0
        d_loss.backward()
        # torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), clip_value=1)
        self.d_optimizer.step()



        '''generator'''
        rllib.basic.pytorch.set_requires_grad(self.discriminator, False)
        self.g_optimizer.zero_grad()
        g_loss = self.calculate_g_loss(data, info)
        g_loss.backward()
        # torch.nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=1)
        self.g_optimizer.step()

        self.writer.add_scalar('loss/d_loss', d_loss.detach().item(), self.step_update)
        self.writer.add_scalar('loss/g_loss', g_loss.detach().item(), self.step_update)

        if self.step_update % self.save_model_interval == 0:
            self._save_model()
        
        return



    def forward(self, data):
        return 0.0

    def calculate_d_loss(self, data):
        return torch.tensor(0.0)
    
    def calculate_g_loss(self, data):
        return torch.tensor(0.0)


    def calculate_gradient_penalty(self, real, fake):
        alpha_shape = torch.tensor(real.shape)
        alpha_shape[1:] = 1
        alpha_shape = list(alpha_shape)
        alpha = torch.rand(alpha_shape, device=self.device)
        interpolates = (alpha * real.data + ((1 - alpha) * fake.data)).requires_grad_(True)

        model_interpolates = self.discriminator(interpolates)
        grad_outputs = torch.ones_like(model_interpolates)

        gradients = grad(
            outputs=model_interpolates, inputs=interpolates, grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) **2)
        return gradient_penalty



class Evaluate(rllib.template.Method):
    dim_noise = 64

    def __init__(self, config: rllib.basic.YamlConfig, writer: rllib.basic.Writer):
        super().__init__(config, writer)

        config.set('method_name', 'WganGp'.upper())
        config.set('dim_noise', self.dim_noise)

        self.discriminator = config.get('net_discriminator', Discriminator)(config).to(self.device)
        self.generator = config.get('net_generator', Generator)(config).to(self.device)
        # self.models_to_load = [self.discriminator, self.generator]
        self.models_to_load = [self.generator]
        self._load_model()
        return



class Generator(rllib.template.Model):
    """
        config: dim_data
    """

    def __init__(self, config):
        super(Generator, self).__init__(config, model_id=0)

        dim_input = config.dim_noise
        dim_output = config.dim_data

        self.fe = FeatureMapper(config, 0, dim_input, dim_output)
        self.apply(rllib.utils.init_weights)
    
    def forward(self, x):
        return self.fe(x)


class Discriminator(rllib.template.Model):
    """
        config: dim_data
    """

    def __init__(self, config):
        super(Discriminator, self).__init__(config, model_id=0)

        dim_input = config.dim_data

        self.fe = FeatureMapper(config, 0, dim_input, 1)
        self.apply(rllib.utils.init_weights)

    def forward(self, x):
        return self.fe(x)



