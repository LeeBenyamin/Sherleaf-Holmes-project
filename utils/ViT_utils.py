import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
import random
from tqdm import tqdm
from torchsummary import summary


def divide_to_patches(image, patch_size=16, grid_w=14, grid_h=14):
    image_width, image_height = image.shape[1], image.shape[2]
    num_of_patches = (image_width // patch_size) * (image_height // patch_size)
    patches_list = []
    patches = np.zeros((num_of_patches, 3, patch_size, patch_size))
    for i in range(grid_h):
        for j in range(grid_w):
            patches_list.append(image[0:3, i*patch_size : (i+1)*patch_size, j*patch_size : (j+1)*patch_size])
    for i in range(len(patches_list)):
        patches[i] = patches_list[i]
    return torch.Tensor(patches)

def flatten_patch_to_sequence(patch):
    flattened_tensor = patch.view(196, -1)
    return flattened_tensor

def show_image_patches(patches: torch.Tensor, grid_width=14, grid_height=14):
    num_images = patches.shape[0]
    fig, axes = plt.subplots(grid_width, grid_height, figsize=(10, 10))
    for i in range(num_images):
        part_of_image = patches[i].clamp(0, 1).numpy()
        ax = axes[i // grid_width, i % grid_height]
        ax.imshow(part_of_image.transpose(1, 2, 0))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_sequenced_patch(patches: torch.Tensor, grid_width=14, grid_height=14):
    num_images = patches.shape[0]
    fig, axes = plt.subplots(1, grid_width * grid_height, figsize=(30, 1))
    for i in range(num_images):
        part_of_image = patches[i].clamp(0, 1).numpy()
        ax = axes[i]
        ax.imshow(part_of_image.transpose(1, 2, 0))
        ax.axis('off')
    plt.show()

