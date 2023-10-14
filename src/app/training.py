import os
import time
import datetime

import torch
from torch.autograd import Variable, grad
from tqdm import tqdm
import torchvision.utils as vutils


def log_progresso(log_file, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"

    with open(log_file, "a", encoding='utf-8') as file:
        file.write(log_entry)


def save_checkpoint(epoch, generator1, generator2, discriminator, optim_g1, optim_g2, optim_d, losses_g1, losses_g2, losses_d, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'generator1_state_dict': generator1.state_dict(),
        'generator2_state_dict': generator2.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g1_state_dict': optim_g1.state_dict(),
        'optimizer_g2_state_dict': optim_g2.state_dict(),
        'optimizer_d_state_dict': optim_d.state_dict(),
        'losses_g1': losses_g1,
        'losses_g2': losses_g2,
        'losses_d': losses_d
    }, path)


def save_generated_samples(generator, noise, sample_dir, epoch, generator_name):
    vutils.save_image(
        generator(noise).data,
        os.path.join(sample_dir, f'fake_samples_{generator_name}_epoch_{epoch:06d}.jpeg'),
        normalize=True
    )


def gradient_penalty(discriminator, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    alpha = alpha.expand_as(real_data)

    interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolated = Variable(interpolated, requires_grad=True).to(device)
    interpolated_prob = discriminator(interpolated)

    gradients = grad(outputs=interpolated_prob, inputs=interpolated,
                     grad_outputs=torch.ones(interpolated_prob.size()).to(device),
                     create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean()


def update_generator(generator, optimizer, z_dim, discriminator, images, device):
    z = Variable(torch.randn(images.size(0), z_dim, 1, 1)).to(device)
    fake_images = generator(z)
    outputs = discriminator(fake_images).squeeze()
    g_loss = -torch.mean(outputs)

    generator.zero_grad()
    g_loss.backward()
    optimizer.step()

    return g_loss


def update_discriminator(discriminator, generator1, generator2, optim_d, z_dim, images, device, lambda_gp):
    z1 = Variable(torch.randn(images.size(0), z_dim, 1, 1)).to(device)
    fake_images_g1 = generator1(z1)

    z2 = Variable(torch.randn(images.size(0), z_dim, 1, 1)).to(device)
    fake_images_g2 = generator2(z2)

    real_prob = discriminator(images)
    fake_prob_g1 = discriminator(fake_images_g1.detach())
    fake_prob_g2 = discriminator(fake_images_g2.detach())

    real_loss = -torch.mean(real_prob)
    fake_loss_g1 = torch.mean(fake_prob_g1)
    fake_loss_g2 = torch.mean(fake_prob_g2)

    # average_fake_images = 0.5 * (fake_images_g1 + fake_images_g2)
    # gp = gradient_penalty(discriminator, images, average_fake_images, device)

    gp = gradient_penalty(
        discriminator, images, fake_images_g1, device
    ) + gradient_penalty( 
        discriminator, images, fake_images_g2, device
    )

    d_loss = real_loss + fake_loss_g1 + fake_loss_g2 + lambda_gp * gp

    discriminator.zero_grad()
    d_loss.backward()
    optim_d.step()

    return d_loss


def train_model(**kwargs):
    fixed_noise = Variable(
        torch.randn(kwargs['sample_size'], kwargs['z_dim'], 1, 1)
    ).to(kwargs['device'])

    for epoch in range(kwargs['last_epoch'], kwargs['num_epochs'] + 1):
        start_time = time.time()

        pbar = tqdm(enumerate(kwargs['data_loader']), total=len(kwargs['data_loader']))
        for _, data in pbar:
            images, _ = data
            images = Variable(images).to(kwargs['device'])

            # Discriminator update
            for _ in range(kwargs['n_critic']):
                d_loss = update_discriminator(
                    kwargs['discriminator'], kwargs['generator1'], kwargs['generator2'],
                    kwargs['optim_d'], kwargs['z_dim'], images, kwargs['device'],
                    kwargs['lambda_gp']
                )

            # Update Generators
            g_loss1 = update_generator(
                kwargs['generator1'], kwargs['optim_g1'], kwargs['z_dim'],
                kwargs['discriminator'], images, kwargs['device']
            )
            g_loss2 = update_generator(
                kwargs['generator2'], kwargs['optim_g2'], kwargs['z_dim'],
                kwargs['discriminator'], images, kwargs['device']
            )

            pbar.set_description(
                f"Epoch {epoch}/{kwargs['num_epochs']}, g_loss1: {g_loss1.data}, "
                f"g_loss2: {g_loss2.data}, d_loss: {d_loss.data}"
            )

        end_time = time.time()
        epoch_duration = end_time - start_time
        log_progresso(
            f"{kwargs['log_dir']}/training.log",
            f"Epoch {epoch}/{kwargs['num_epochs']}, "
            f"g_loss1: {g_loss1.data}, g_loss2: {g_loss2.data}, "
            f"d_loss: {d_loss.data}, "
            f"Time: {epoch_duration:.2f} seconds"
        )

        kwargs['losses_g1'].append(g_loss1.data.cpu())
        kwargs['losses_g2'].append(g_loss2.data.cpu())
        kwargs['losses_d'].append(d_loss.data.cpu())

        save_generated_samples(kwargs['generator1'], fixed_noise, kwargs['sample_dir'], epoch, "g1")
        save_generated_samples(kwargs['generator2'], fixed_noise, kwargs['sample_dir'], epoch, "g2")

        if (epoch) % kwargs['save_model_at'] == 0:
            save_checkpoint(
                epoch,
                kwargs['generator1'],
                kwargs['generator2'],
                kwargs['discriminator'],
                kwargs['optim_g1'],
                kwargs['optim_g2'],
                kwargs['optim_d'],
                kwargs['losses_g1'],
                kwargs['losses_g2'],
                kwargs['losses_d'],
                f"{kwargs['weights_path']}/checkpoint.pth"
            )

    return kwargs['losses_g1'], kwargs['losses_g2'], kwargs['losses_d']
