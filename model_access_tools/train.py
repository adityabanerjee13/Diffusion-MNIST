import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader

from Unet.model_unet import Unet
from Scheduler.noise_schedule import NoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(arg):
    # Read config file
    with open(arg.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Create NoiseScheduler
    scheduler = NoiseScheduler(diffusion_config, device)
    
    # Create the dataset
    mnist = MnistDataset('train', im_path=dataset_config['im_path'])
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle = True, num_workers = 4)

    # Instanciate the model
    model = Unet(model_config).to(device)
    model.train()

    # Output directory
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'], train_config['ckpt_name'])):
        print('Checkpoint Found : loading checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), 
                                                      map_location=device))
        
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr = train_config['lr'])
    criteria = torch.nn.MSELoss()

    # Run training
    for i in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],))#
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t.to(device))

            loss = criteria(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(i+1, np.mean(losses)))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name']))
        
    print("Training Complete")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training diffusion model')
    parser.add_argument('--config', dest='config_path',
                        default='Config/ddpm_config.yaml', type=str)
    arg = parser.parse_args()
    train(arg)