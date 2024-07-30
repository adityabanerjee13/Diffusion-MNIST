import torch
import numpy as np

class NoiseScheduler:
    def __init__(self, diffusion_config, device):

        num_timesteps=diffusion_config['num_timesteps']
        beta_start=diffusion_config['beta_start']
        beta_end=diffusion_config['beta_end']
        sampling_scheme=diffusion_config['sampling_scheme']
        eta=diffusion_config['eta']
        self.device = device

        self.sampling_scheme = sampling_scheme
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = (1 - self.betas).to(device)
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod).to(device)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod).to(device)


    def add_noise(self, orignal, noise, t):
        orignal_shape = orignal.shape
        batch_size = orignal_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        for _ in range(len(orignal_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod*orignal + sqrt_one_minus_alpha_cum_prod*noise
    
    def sample_prev_timestep(self, xt_i, noise_pred, t_i_minus_1, t_i = None):
        x0 = (xt_i - (self.sqrt_one_minus_alpha_cum_prod[t_i] * noise_pred))\
             / self.sqrt_alpha_cum_prod[t_i]
        x0 = torch.clamp(x0, -1, 1)

        if t_i_minus_1==0:
            mean = xt_i - ((self.betas[t_i] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t_i]))
            mean = mean / torch.sqrt(self.alphas[t_i])
            return mean, x0
        elif self.sampling_scheme == 'DDPM':
            mean = xt_i - ((self.betas[t_i] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t_i]))
            mean = mean / torch.sqrt(self.alphas[t_i])
            variance = (1 - self.alpha_cum_prod[t_i - 1]) / (1.0 - self.alpha_cum_prod[t_i])
            variance = variance * self.betas[t_i]
            sigma = variance ** 0.5
            z = torch.randn_like(xt_i).to(self.device)
            return mean + sigma * z, x0
        elif self.sampling_scheme == 'DDIM':
            # required variables for DDIM scheme
            sigma_t_i = self.eta * (self.sqrt_one_minus_alpha_cum_prod[t_i_minus_1]/self.sqrt_one_minus_alpha_cum_prod[t_i]) \
                * torch.sqrt(1 - (self.alpha_cum_prod[t_i]/self.alpha_cum_prod[t_i_minus_1]))
            # initialize stochastic component
            z = torch.randn_like(xt_i).to(self.device)
            # calculating previous step
            xt_i_minus_1 = self.sqrt_alpha_cum_prod[t_i_minus_1] * x0 \
                + torch.sqrt(1 - self.alpha_cum_prod[t_i_minus_1] - sigma_t_i**2) * noise_pred \
                + sigma_t_i * z
            return xt_i_minus_1, x0

class ScoreNoiseScheduler:
    def __init__(self, diffusion_config, device):

        num_timesteps=diffusion_config['num_timesteps']
        sigma_1=diffusion_config['sigma_1']
        sigma_L=diffusion_config['sigma_L']
        self.epsilon=diffusion_config['epsilon']
        self.device = device

        self.num_timesteps = num_timesteps
        c = (sigma_L/sigma_1)**(1/(num_timesteps-1))
        self.sigmas = torch.tensor([sigma_1*(c**x) for x in range(num_timesteps)]).to(device)
        self.alphas = ((self.sigmas/sigma_L)**2*self.epsilon).to(device)


    def add_noise(self, orignal, noise, t):
        orignal_shape = orignal.shape
        batch_size = orignal_shape[0]

        sigmas = self.sigmas[t].reshape(batch_size)

        for _ in range(len(orignal_shape)-1):
            sigmas = sigmas.unsqueeze(-1)

        return orignal + sigmas*noise, -noise/sigmas
    
    def langevin_step(self, xt, noise_pred, i):
        z = torch.randn_like(xt).to(self.device)
        xt_minus_1 = xt + (self.alphas[i]/2) * noise_pred + torch.sqrt(self.alphas[i]) * z
        
        return xt_minus_1
        