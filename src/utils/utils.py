import os
import json
import datetime

import torch
from torchvision import transforms, datasets
import torch.nn as nn


def create_next_version_directory(base_dir, continue_last_training):
    versions = [d for d in os.listdir(base_dir) if d.startswith('v') and os.path.isdir(os.path.join(base_dir, d))]

    if not versions:
        next_version = 1
    else:
        if continue_last_training:
            return f"v{max(int(v[1:]) for v in versions)}"

        next_version = max(int(v[1:]) for v in versions) + 1

    new_dir_base = os.path.join(base_dir, f'v{next_version}')

    for sub_dir in ['', 'samples', 'weights', 'log']:
        os.makedirs(os.path.join(new_dir_base, sub_dir), exist_ok=True)

    return f"v{next_version}"


def check_if_gpu_available():
    print('Use GPU:', torch.cuda.is_available())

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPUs available: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}")
    else:
        print("No GPU available.")


def check_if_set_seed(seed=None):
    if seed:
        torch.manual_seed(seed)
        print(f'Using the Seed: {seed}')
    else:
        print('Using random seed.')


def get_params(path_file):
    with open(path_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return params


def print_datetime(label="Current Date and Time"):
    data_hora_atual = datetime.datetime.now()
    data_hora_formatada = data_hora_atual.strftime("%d/%m/%Y %H:%M:%S")
    print(f'\n{label}: {data_hora_formatada}')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def dataloader(directory, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(directory, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def create_train_config(
        params,
        generator1,
        generator2,
        discriminator,
        optim_g1,
        optim_g2,
        optim_d,
        data_loader,
        last_epoch,
        losses_g1,
        losses_g2,
        losses_d,
        data_dir,
        device
    ):

    training_config = {
        'generator1': generator1,
        'generator2': generator2,
        'discriminator': discriminator,
        'weights_path': os.path.join(data_dir, 'weights'),
        'n_critic': params["n_critic"],
        'sample_size': params["sample_size"],
        'sample_dir': os.path.join(data_dir, 'samples'),
        'optim_g1': optim_g1,
        'optim_g2': optim_g2,
        'optim_d': optim_d,
        'data_loader': data_loader,
        'device': device,
        'z_dim': params["z_dim"],
        'lambda_gp': params["lambda_gp"],
        'num_epochs': params["num_epochs"],
        'last_epoch': last_epoch,
        'save_model_at': params['save_model_at'],
        'log_dir': os.path.join(data_dir, 'log'),
        'losses_g1': losses_g1,
        'losses_g2': losses_g2,
        'losses_d': losses_d
    }

    return training_config
