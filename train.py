import argparse
import utils

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image

parser = argparse.ArgumentParser(description='Train')

# Add command line arguments
parser.add_argument('data_directory', nargs='*', action="store", default="./flowers/")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)

args = parser.parse_args()

data_dir = args.data_directory
save_dir = args.save_dir
lr = args.learning_rate
arch = args.arch
epochs = args.epochs

train_dataloader, val_dataloader, test_dataloader = utils.load_data(data_dir = data_dir)
model, criterion, optimizer = utils.build_nn(arch=arch, lr=lr)
utils.train_nn(model, criterion, optimizer, train_dataloader, val_dataloader,epochs=epochs)
utils.save_checkpoint(model, optimizer, path=save_dir, data_dir=data_dir, epochs=epochs, lr=lr)

print('All done!')