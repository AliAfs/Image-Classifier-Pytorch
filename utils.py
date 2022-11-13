import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image

def get_datasets(data_dir='./flowers'):
    '''
    The function applies required transformations and create torch datasets for training, validation and tests sets
    
    Arguments:
                data_dir: (string) data direvtory
                
    Return:
                train_image_dataset, val_image_dataset, test_image_dataset
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Apply different transformations for the train, validation and test sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_image_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_image_dataset = datasets.ImageFolder(valid_dir, transform=val_transforms)
    test_image_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return train_image_dataset, val_image_dataset, test_image_dataset

def load_data(data_dir = './flowers'):
    '''
    The function loads the train, validation and test data from given directory and return dataloaders for corresponding sets.
    
    Arguments:  
                data_dir: data folder's path
    Return:
                train_dataloader
                val_dataloader
                test_dataloader
    '''
    # Get data sets with applied transformations
    train_image_dataset, val_image_dataset, test_image_dataset = get_datasets(data_dir)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_image_dataset, batch_size=64, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_image_dataset, batch_size=64, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

def build_nn(arch='vgg16', lr=0.001):
    '''
    The function builds the NN architecture
    
    Arguments:
                arch: architecture for network
                lr: learning rate
    Return:
                model
                criterion
                optimizer
    '''
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print('Supported model [vgg16]')
        
    for param in model.parameters():
        param.requires_grad = False
    
    # Define classifier
    classifier = nn.Sequential(nn.Linear(25088, 512),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(512, 102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    return model, criterion, optimizer

def train_nn(model, criterion, optimizer, train_dataloader, val_dataloader, epochs=5):
    '''
    The function trains nn model with given criterion and optimizer
    
    Arguments:
                model: Neural Network model
                criterion: loss function
                optimizer
                train_dataloader: data loader for training set
                epoch
                
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    running_loss = 0

    print('----------------------Training is starting----------------------')
    for epoch in range(epochs):
        for inputs, labels in train_dataloader:
            # Move inputs and labels tensors to the device(gpu, cpu)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forwardpass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # Backwardpass and optimize parameters
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_accuracy = 0
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dataloader:

                inputs, labels = inputs.to(device), labels.to(device)

                logps = model.forward(inputs)
                
                val_loss += criterion(logps, labels).item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Running loss: {running_loss/len(train_dataloader):.3f}.. "
                    f"Validation loss: {val_loss/len(val_dataloader):.3f}.. "
                    f"Validation accuracy: {val_accuracy/len(val_dataloader):.3f}")

        running_loss = 0
        model.train()
        
def save_checkpoint(model, optimizer, path='checkpoint.pth', data_dir='./flowers',epochs=5, lr=0.001):
    '''
    The function saves the model checkpoint to the given path
    '''
    
    train_image_dataset, _, _ = get_datasets(data_dir)
    model.class_to_idx = train_image_dataset.class_to_idx

    state = {'architecture': 'vgg16',
             'classifier': model.classifier,
             'epoch': epochs,
             'lr': lr,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'class_to_idx': model.class_to_idx
            }

    torch.save(state, path)
    
def load_checkpoint(path='checkpoint.pth'):
    '''
    Function loads deep learning model from checkpoint path
    '''
    
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    # load the checkpoint from given path
    checkpoint = torch.load(path, map_location=map_location)
    
    # download pretrained model
    model = models.vgg16(pretrained=True);
    
    for param in model.parameters(): param.requires_grad = False
    
    # load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    
    # resize the image
    width, height = image.size   # Get dimensions
    if width > height:
        image.resize((width*256//height,256))
    else:
        image.resize((256,height*256//width))    
    
    new_width = 224
    new_height = 224
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    # normalize color channels
    np_image = np_image.astype(float)
    np_image *= 1.0/255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean) / std
    
    # change dimension
    np_image = np_image.transpose((2,0,1))
    
    return np_image

def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cpu')
    model.eval()
    
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze_(0)
    
    logps = model.forward(img)
    ps = torch.exp(logps)
    
    probs, classes = ps.topk(topk)
    # convert probs, classes tensors to list 
    probs = probs.tolist()[0]
    classes = classes.tolist()[0]
    
    return probs, classes