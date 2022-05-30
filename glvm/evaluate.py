import rllib

import time

import torch


class Evaluate(rllib.template.MethodSingleAgent):
    def __init__(self, config, writer):
        super(Evaluate, self).__init__(config, writer)

        self.model_dir = config.model_dir
        self.model_num = config.model_num

        method_name = self.model_dir.split('/')[-3].split('-')[0]
        config.set('method_name', method_name)

        if method_name == 'VanillaVAE':
            from . import vanilla_vae
            self.model: vanilla_vae.Model = vanilla_vae.Model(config).to(self.device)
            self.models_to_load = [self.model]
        elif method_name == 'AutoEncoder':
            from . import auto_encoder
            self.model: auto_encoder.Model = auto_encoder.Model(config).to(self.device)
            self.models_to_load = [self.model]
        else:
            raise NotImplementedError('No such method: ' + str(method_name))
        
        self._load_model()
        return
    

    def store(self, experience):
        return


    @torch.no_grad()
    def select_action(self, _):
        super().select_action()
        action = torch.Tensor(1,self.dim_action).uniform_(-1,1)
        return action





class Evaluate(object):
    method_cls = None

    def __init__(self, config: rllib.basic.YamlConfig, writer: rllib.basic.Writer):
        method_cls = config.get('method_cls', self.method_cls)
        self.method = method_cls(config, writer)
        self.method._load_model()

        for key, value in self.method.__class__.__dict__.items():
            if not callable(value) and not hasattr(value, '__get__'):
                setattr(self, key, value)
        for key, value in self.method.__dict__.items():
            if not callable(value) and not hasattr(value, '__get__'):
                setattr(self, key, value)
        return


    def update_parameters_(self):
        t1 = time.time()
        for i, data_samples in enumerate(self.evaluate_dataloader):
            print('    evaluate ratio: {} / {}'.format(i, len(self.evaluate_dataset)/self.batch_size), end='\r', flush=True)
            with torch.no_grad():
                self.method.evaluate_parameters(rllib.basic.Data(**data_samples).to(self.device))
        t2 = time.time()
        print('\nevaluate one epoch time: ', t2-t1, 's')
        return

