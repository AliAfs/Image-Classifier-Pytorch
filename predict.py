import argparse
import utils
import json

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image

parser = argparse.ArgumentParser(description='Predict')

# Add command line arguments
parser.add_argument('input_img', nargs='*', action="store", default="./flowers/test/100/image_07897.jpg")
parser.add_argument('checkpoint', nargs='*', action="store", default="checkpoint.pth")
parser.add_argument('--top_k', dest="top_k", action="store", default=3, type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

args = parser.parse_args()

input_img = args.input_img
checkpoint = args.checkpoint
top_k = args.top_k
cat_to_name = args.category_names


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# load checkponit
model = utils.load_checkpoint(checkpoint)
probs, classes = utils.predict(input_img, model, topk=top_k)
probs_round = [round(prob, 3) for prob in probs]

# match class to idx
class_names = []
for class_ in classes:
    for key, value in model.class_to_idx.items():
        if value == class_:
            class_names.append(cat_to_name[key])
            
print(f'Top {top_k} classes and corresponding probabilities:')
for class_name, prob in zip(class_names, probs_round):
    print('Class: {} --- Probability: {}'.format(class_name, prob))