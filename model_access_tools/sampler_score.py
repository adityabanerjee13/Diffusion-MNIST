import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision.utils import make_grid
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader

from Unet.model_unet import Unet
from Scheduler.noise_schedule import NoiseScheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sampler(model, scheduler, config):

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    Sample_PATH = os.path.join(train_config['task_name'], 'samples')
    if not os.path.exists(Sample_PATH):
        os.mkdir(Sample_PATH)

    # time shedule according to sampling scheme
    if diffusion_config['sampling_scheme'] == 'DDPM':
        t = np.linspace(diffusion_config['num_timesteps']-1, 0, diffusion_config['num_timesteps']).astype(int)
    elif diffusion_config['sampling_scheme'] == 'DDIM':
        t = np.linspace(diffusion_config['num_timesteps']-1, 0, diffusion_config['sample_timesteps']).astype(int)

    # initialization 
    t_i = t[0]
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size']
                      )).to(device)
    
    for t_i_minus_1 in tqdm(t[1:]):
        # Get prediction of noise
        print(t_i_minus_1.shape)
        noise_pred = model(xt, torch.as_tensor(t_i_minus_1).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt_i-1
        xt_i_minus_1, _ = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t_i_minus_1), torch.as_tensor(t_i))

        # Save x0
        if t_i_minus_1%2 == 0:
            ims = torch.clamp(xt_i_minus_1, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            grid = make_grid(ims, nrow=train_config['num_grid_rows'])
            img  = torchvision.transforms.ToPILImage()(grid)
            img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(t_i_minus_1)))
            img.close()

        xt = xt_i_minus_1
        t_i = t_i_minus_1



def infer(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Create NoiseScheduler
    scheduler = NoiseScheduler(diffusion_config, device)
    
    # Instanciate the model
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()

    # Create the noise scheduler
    with torch.no_grad():
        sampler(model, scheduler, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for diffusion image generation')
    parser.add_argument('--config', dest='config_path', 
                        default='Config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)