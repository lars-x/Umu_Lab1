import numpy as np
import matplotlib.pyplot as plt

import os
import urllib.request
import zipfile
import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image

data_dir = './data/'
training_file = data_dir + 'train.p'
validation_file = data_dir + 'valid.p'
testing_file = data_dir + 'test.p'
zip_file = data_dir + 'traffic-signs-data.zip'

if not (os.path.exists(training_file) and
        os.path.exists(validation_file) and
        os.path.exists(testing_file)):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    print('Beginning data file downloading')
    url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip'
    urllib.request.urlretrieve(url, zip_file)

    print('Beginning file unzip')
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(data_dir)
    zip_ref.close()
    os.listdir(data_dir)
    print('Done')
else:
    print('No data file downloading needed')

print('Load pickled data')
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test,  y_test = test['features'], test['labels']

# 1.2 Dataset Summary & Exploration

print("Number of training examples   =", X_train.shape[0])
print("Number of validation examples =", X_valid.shape[0])
print("Number of testing examples    =", X_test.shape[0])
print("Total examples                =", X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
# What's the shape of an traffic sign image?
print("Image data shape  =", X_train.shape[1:])
# How many unique classes/labels there are in the dataset.
print("Number of classes =", len(np.unique(y_train)))

# Visualize Image functions


def show_image(image, title):
    if len(image.shape) <= 3:
        plt.imshow(image, cmap="gray")
    elif image.shape[2] == 1:  # Works localy, but not in Colab
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.axis('off')
    plt.title(title)
    plt.show()


def show_images(X, y, indexes, cols):
    n = len(indexes)
    rows = int(np.ceil(n / cols))
    fig = plt.figure()
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i+1)
            plt.axis("off")
            if i >= n:
                continue
            index = indexes[i]
            plt.imshow(X[index].squeeze())
            plt.title(f'Class = {y[index]}')
            i = i + 1

    # fig.tight_layout()
    fig.set_figheight(rows*2)
    fig.set_figwidth(12)
    plt.show()


def get_n_random_indices_for_class(y, n_random_indices, classe):
    indexes = []
    for index, y in enumerate(y_train):
        if y == classe:
            indexes.append(index)

    indexes_total = len(indexes)
    indexes_random = np.random.choice(indexes, n_random_indices, replace=False)
    return indexes_random, indexes_total


def plot_image_distribution(x, y, xlabel, ylabel, width, color):
    plt.figure(figsize=(15, 7))
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.bar(x, y, width, color=color)
    plt.show()


print('Show some traffic signs')
i = 11137
show_image(X_train[i], f'Image {i}')
i = 29730
show_image(X_train[i], f'Image {i}')
print(f'Stop Class Id = {y_train[i]}')

print('Show some Stop signs')
indexes = [29895, 29375, 29459, 29624, 29506, 29343]
show_images(X_train, y_train, indexes, cols=6)

print('Show some random Stop signs')
indexes_random, indexes_total = get_n_random_indices_for_class(y_train, 18, 14)
show_images(X_train, y_train, indexes_random, cols=6)

print('Show traffic signs distibution')
classes, counts = np.unique(y_train, return_counts=True)
plot_image_distribution(classes, counts, 'Classes', '# Training Examples', 0.7, 'blue')

# 1.3 Create a Image Pytorch Dataset & DataLoader:s


class ImageDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor()])

train_dataset = ImageDataset(X_train, y_train, transform=transform)
valid_dataset = ImageDataset(X_valid, y_valid, transform=transform)
test_dataset = ImageDataset(X_test,  y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

print('Some ImageDataset & DataLoader tests...')


def number_of_batches(loader):
    n = 0
    for batch_idx, batch in enumerate(loader):
        n = n + 1
    return n


# TODO: Very slow after transforms.ToPILImage() was added
print('train_loader =', number_of_batches(train_loader))
print('valid_loader =', number_of_batches(valid_loader))
print('test_loader  =', number_of_batches(test_loader))


def tensor_image_to_image(tensor_image):
    # From https://stackoverflow.com/questions/64629702/pytorch-transform-totensor-changes-image
    image = np.moveaxis(tensor_image.numpy()*255, 0, -1).astype(np.uint8)

    # From: https://stackoverflow.com/questions/54664329/invalid-dimension-for-image-data-in-plt-imshow
    if image.shape[2] == 1:
        image = np.squeeze(image, axis=-1)

    return image


def show_first_image_in_some_batch(loader):
    for batch_idx, batch in enumerate(loader):
        ## print("\nBatch = " + str(batch_idx))
        if (batch_idx >= 42):
            X = batch[0]
            y = batch[1]
            tensor_image = X[0]
            y_c = y[0]
            image = tensor_image_to_image(tensor_image)
            show_image(image, str(y_c.numpy()))
            break


show_first_image_in_some_batch(train_loader)
show_first_image_in_some_batch(train_loader)
show_first_image_in_some_batch(valid_loader)
show_first_image_in_some_batch(test_loader)

# 1.4 Build a CNN model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


model_cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)


# 1.5 Explore the model

def exlorer(loader, model):
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        print*loss()
        break


exlorer(train_loader, model_cnn)

# 1.6 Train the model


def epoch(loader, model, opt=None):
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)
