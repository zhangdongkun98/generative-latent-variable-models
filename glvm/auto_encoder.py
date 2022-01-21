import rllib

from typing import List

import torch
import torch.nn as nn
from torch.optim import Adam


class AutoEncoder(rllib.template.MethodSingleAgent):
    lr_model = 0.0003

    buffer_size = 10000
    batch_size = 144
    weight = batch_size / buffer_size

    start_timesteps = 10000
    
    save_model_interval = 200

    def __init__(self, config: rllib.basic.YamlConfig, writer: rllib.basic.Writer):
        '''
        '''

        super().__init__(config, writer)

        self.model = Model(config).to(self.device)
        self.models_to_save = [self.model]

        self.optimizer = Adam(self.model.parameters(), lr=self.lr_model)
        self.model_loss = nn.MSELoss()
        self._memory = config.get('buffer', ReplayBuffer)(self.buffer_size, self.batch_size, self.device)
        return


    def update_parameters(self):
        if len(self._memory) < self.start_timesteps:
            return
        self.update_parameters_start()

        '''load data batch'''
        experience = self._memory.sample()
        input: torch.Tensor = experience.input

        output = self.model(input)
        loss = self.model_loss(output, input.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('loss/loss', loss.detach().item(), self.step_update)

        if self.step_update % self.save_model_interval == 0: self._save_model()
        return


    @torch.no_grad()
    def select_action(self, _):
        super().select_action()
        action = torch.Tensor(1,self.dim_action).uniform_(-1,1)
        return action


class Model(rllib.template.Model):
    def __init__(self, config):
        super(Model, self).__init__(config)

        self.dim_latent = config.dim_latent

        self.encoder = nn.Sequential(
            nn.Conv2d(config.in_channels, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(),
        )
        self.encoder_output = nn.Sequential(
            nn.Linear(512 *4, config.dim_latent), nn.Tanh()
        )

        self.decoder_input = nn.Linear(config.dim_latent, 512 *4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.ConvTranspose2d(32, config.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh(),
        )
        self.apply(rllib.utils.init_weights)
        return


    def forward(self, input: torch.Tensor):
        z = self.encode(input)
        output = self.decode(z)
        return output


    def encode(self, input: torch.Tensor):
        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)
        return self.encoder_output(x)

    def decode(self, z: torch.Tensor):
        x = self.decoder_input(z)
        x = x.view(z.shape[0], 512, 2, 2)
        x = self.decoder(x)
        return x



class ReplayBuffer(rllib.buffer.ReplayBuffer):
    def _batch_stack(self, batch):
        image = torch.cat(list(batch), dim=0)
        experience = rllib.template.Experience(input=image)
        return experience

