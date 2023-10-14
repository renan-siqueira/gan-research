import time
import os

import torch
from torch import optim

from src.app.g1 import Generator as Generator_g1
from src.app.g2 import Generator as Generator_g2
from src.app.discriminator import Discriminator
from src.app.training import train_model
from src.settings import settings
from src.utils import utils


def load_checkpoint(path, generator1, generator2, discriminator, optim_g1, optim_g2, optim_d):
    if not os.path.exists(path):
        print("No checkpoint found.")
        return 1, [], [], []

    checkpoint = torch.load(path)

    generator1.load_state_dict(checkpoint['generator1_state_dict'])
    generator2.load_state_dict(checkpoint['generator2_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optim_g1.load_state_dict(checkpoint['optimizer_g1_state_dict'])
    optim_g2.load_state_dict(checkpoint['optimizer_g2_state_dict'])
    optim_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    epoch = checkpoint['epoch'] + 1
    losses_g1 = checkpoint['losses_g1']
    losses_g2 = checkpoint['losses_g2']
    losses_d = checkpoint['losses_d']

    print(f'Checkpoint loaded, starting from epoch {epoch}')
    return epoch, losses_g1, losses_g2, losses_d


def setup_training(params):
    utils.check_if_gpu_available()
    utils.check_if_set_seed(params["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator1 = Generator_g1(
        params["z_dim"], params["channels_img"], params["features_g"], img_size=params['image_size']
    ).to(device)
    generator1.apply(utils.weights_init)

    generator2 = Generator_g2(
        params["z_dim"], params["channels_img"], params["features_g"], img_size=params['image_size']
    ).to(device)
    generator2.apply(utils.weights_init)

    discriminator = Discriminator(
        params["channels_img"], params["features_d"], params["alpha"], img_size=params['image_size']
    ).to(device)
    discriminator.apply(utils.weights_init)

    optim_g1 = optim.Adam(
        generator1.parameters(), lr=params["lr_g"],
        betas=(params['g_beta_min'], params['g_beta_max'])
    )
    optim_g2 = optim.Adam(
        generator2.parameters(), lr=params["lr_g"],
        betas=(params['g_beta_min'], params['g_beta_max'])
    )
    optim_d = optim.Adam(
        discriminator.parameters(), lr=params["lr_d"],
        betas=(params['d_beta_min'], params['d_beta_max'])
    )

    return generator1, generator2, discriminator, optim_g1, optim_g2, optim_d, device


def setup_directories_and_checkpoint(path_data, params, generator1, generator2, discriminator, optim_g1, optim_g2, optim_d):
    training_version = utils.create_next_version_directory(
        path_data, params['continue_last_training']
    )
    data_dir = os.path.join(path_data, training_version)
    print('Training version:', training_version)

    last_epoch, losses_g1, losses_g2, losses_d = load_checkpoint(
        os.path.join(data_dir, 'weights', 'checkpoint.pth'),
        generator1, generator2, discriminator, optim_g1, optim_g2, optim_d
    )

    return data_dir, last_epoch, losses_g1, losses_g2, losses_d


def main(params, path_data, path_dataset):
    time_start = time.time()
    utils.print_datetime()

    generator1, generator2, discriminator, optim_g1, optim_g2, optim_d, device = setup_training(
        params
    )
    data_loader = utils.dataloader(path_dataset, params["image_size"], params["batch_size"])

    data_dir, last_epoch, losses_g1, losses_g2, losses_d = setup_directories_and_checkpoint(
        path_data, params, generator1, generator2, discriminator, optim_g1, optim_g2, optim_d
    )

    training_config = utils.create_train_config(
        params, generator1, generator2, discriminator, optim_g1, optim_g2, optim_d,
        data_loader, last_epoch, losses_g1, losses_g2, losses_d, data_dir, device
    )

    train_model(**training_config)

    time_end = time.time()
    time_total = (time_end - time_start) / 60

    print(f"The code took {round(time_total, 1)} minutes to execute.")
    utils.print_datetime()


if __name__ == '__main__':
    PARAMS = utils.get_params(settings.PATH_PARAMS)
    main(PARAMS, settings.PATH_DATA, settings.PATH_DATASET)
