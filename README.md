# GAN Research with Dual Generators

This project presents a novel approach to Generative Adversarial Networks (GANs) by employing two generators in competition with each other and against a common discriminator.

## Overview

Traditional GANs use a single generator and discriminator to learn and generate realistic images. This project explores the use of two  generators that both compete against a common discriminator. The aim is to study the performance and characteristics of dual generators in GAN training.

## Features

- Dual Generator Architecture.
- Use of gradient penalty for improved training stability.
- Dynamic directory creation for model checkpoints and generated samples.
- Configurable parameters through a JSON file.
- GPU support detection.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- tqdm

### Installation

1. Clone the repository.
2. Install the required packages.

[*See my other projects for more details about setup and configuration](https://github.com/renan-siqueira/my-own-WGAN-GP-implementation)

### Usage

1. Update the `src/settings/settings.py` with the correct paths.

2. Configure the training parameters in `src/json/params.json`.

3. Execute the training:

```bash
python run.py
```

## Structure

- `run.py`: Entry point for training.
- `src/app/training.py`: Contains training-related functions.
- `src/utils/utils.py`: Utility functions.
- `src/json/params.json`: Training parameters in JSON format.
- `src/settings/settings.py`: Path settings.

## License

This project is licensed under the MIT License.
