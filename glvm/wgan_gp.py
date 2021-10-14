import rllib
from rllib.template.model import FeatureMapper

import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# class GradientPaneltyLoss(nn.Module):
#     def __init__(self):
#          super(GradientPaneltyLoss, self).__init__()

#     def forward(self, y, x):
#         """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
#         weight = torch.ones_like(y)
#         dydx = torch.autograd.grad(outputs=y,
#                                    inputs=x,
#                                    grad_outputs=weight,
#                                    retain_graph=True,
#                                    create_graph=True,
#                                    only_inputs=True)[0]

#         dydx = dydx.view(dydx.size(0), -1)
#         dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
#         return torch.mean((dydx_l2norm - 1) ** 2)




class WGANGP(rllib.template.Method):
    lr_g = 1e-4
    lr_d = 1e-4
    weight_decay = 5e-4
    batch_size = 32

    dim_noise = 64

    num_workers = 16

    save_model_interval = 2000

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
    

        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.rand((self.batch_size, 1), device=self.device)
        # Get random interpolation between real and fake data
        interpolates = (alpha * real.data + ((1 - alpha) * fake.data)).requires_grad_(True)

        model_interpolates = self.discriminator(interpolates)
        # grad_outputs = torch.ones(model_interpolates.size(), device=self.device, requires_grad=False)
        # grad_outputs = torch.ones_like(model_interpolates, requires_grad=False)
        grad_outputs = torch.ones_like(model_interpolates)

        # Get gradient w.r.t. interpolates
        gradients = grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) **2)
        return gradient_penalty




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


