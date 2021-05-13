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

# ## 1.2 Dataset Summary & Exploration

print("Number of training examples   =", X_train.shape[0])
print("Number of validation examples =", X_valid.shape[0])
print("Number of testing examples    =", X_test.shape[0])
print("Total examples                =", X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
# What's the shape of an traffic sign image?
print("Image data shape  =", X_train.shape[1:])
number_of_different_traffics_signs = len(np.unique(y_train))
print("Number of classes =", number_of_different_traffics_signs)

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


def get_n_random_indices_for_class(ys, n_random_indices, classe):
    indexes = []
    for index, y in enumerate(ys):
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
show_image(X_train[i], f'A scary monster {i}')
i = 29730
show_image(X_train[i], f'Image {i}')
print(f'Stop Class Id = {y_train[i]}')


print('Show some Stop signs')
indexes = [29895, 29375, 29459, 29624, 29506, 29343]
show_images(X_train, y_train, indexes, cols=6)


print('Show some random Stop signs')
stop_sign_class_id = 14
indexes_random, indexes_total = get_n_random_indices_for_class(y_train, 18, stop_sign_class_id)
show_images(X_train, y_train, indexes_random, cols=6)


print('Show traffic signs distibution')
classes, counts = np.unique(y_train, return_counts=True)
plot_image_distribution(classes, counts, 'Classes', '# Training Examples', 0.7, 'blue')

# ## 1.3 Create Image Pytorch Datasets & DataLoaders


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


def show_first_image_in_some_batch(loader, batch_number=42):
    for batch_idx, batch in enumerate(loader):
        ## print("\nBatch = " + str(batch_idx))
        if (batch_idx >= batch_number):
            X = batch[0]
            y = batch[1]
            tensor_image = X[0]
            y_c = y[0]
            image = tensor_image_to_image(tensor_image)
            print(image.shape)
            show_image(image, str(y_c.numpy()))
            break


show_first_image_in_some_batch(train_loader, batch_number=42)
show_first_image_in_some_batch(train_loader, batch_number=42)
show_first_image_in_some_batch(valid_loader, batch_number=42)
show_first_image_in_some_batch(test_loader, batch_number=42)

# ## 1.4 Build a CNN model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


number_of_classes = number_of_different_traffics_signs
model_cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(4096, 100), nn.ReLU(),
                          nn.Linear(100, number_of_classes)).to(device)

# ## 1.5 Explore the model

print(model_cnn.parameters)


def exlorer(loader, model):
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        print(X.shape)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        print(loss.item())
        break


exlorer(train_loader, model_cnn)

# ## 1.6 Train the model


def epoch(loader, model, opt=None):
    total_error = 0.0
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            # In PyTorch, we need to set the gradients to zero before starting to do
            # backpropragation because PyTorch accumulates the gradients on subsequent
            # backward passes.
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_error += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

    n = len(loader.dataset)
    epoc_error = total_error / n
    epoc_loss = total_loss / n

    return epoc_error, epoc_loss


ts = []
train_errors = []
train_losses = []
valid_errors = []
valid_losses = []

max_epochs = 10
opt = optim.SGD(model_cnn.parameters(), lr=0.1)
for t in range(max_epochs):
    train_error, train_loss = epoch(train_loader, model_cnn, opt)
    valid_error, valid_loss = epoch(valid_loader, model_cnn)
    if t == 4:
        for param_group in opt.param_groups:
            param_group["lr"] = 0.01

    print(f'{t+1}\t{train_error:.6f}\t{train_loss:.6f}\t{valid_error:.6f}\t{valid_loss:.6f}')
    ts.append(t+1)
    train_errors.append(train_error)
    train_losses.append(train_loss)
    valid_errors.append(valid_error)
    valid_losses.append(valid_loss)

print(f'Final Train accuracy      = {100*(1-train_error):.2f}%')
print(f'Final Validation accuracy = {100*(1-valid_error):.2f}%')

# ## 1.7 Check overfitting


plt.plot(ts, train_errors, '-b')
plt.plot(ts, valid_errors, '-r')
plt.grid()
plt.show()

# ## 1.8 Calculate the accuracy on the testset


def accuracy(loader, model):
    total_accuracy = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        total_accuracy += (yp.max(dim=1)[1] == y).sum().item()

    n = len(loader.dataset)
    epoc__accuracy = total_accuracy / n
    return epoc__accuracy


print(f'Train accuracy       = {accuracy(train_loader, model_cnn)*100:.2f}%')
print(f'Validation accuracy  = {accuracy(valid_loader, model_cnn)*100:.2f}%')
print(f'Test accuracy        = {accuracy(test_loader,  model_cnn)*100:.2f}%')

# # 2 Attack!
# ## 2.-1 Table of ClassId & SignName

# ## 2.0 Init stuff

# See table above
stop_sign_class_id = 14
speed_limit_50_class_id = 2


def get_indices_for_class(ys, classe):
    indexes = []
    for index, y in enumerate(ys):
        if y == classe:
            indexes.append(index)

    return indexes


indexes_stop_sign = get_indices_for_class(y_test, stop_sign_class_id)
X_stop_sign = X_test[indexes_stop_sign]
n = len(indexes_stop_sign)
y_stop_sign = np.ones(n)*stop_sign_class_id

stop_sign_dataset = ImageDataset(X_stop_sign, y_stop_sign, transform=transform)
stop_sign_loader = DataLoader(stop_sign_dataset, batch_size=stop_sign_dataset.__len__(), shuffle=False)

stop_sign_dataset = ImageDataset(X_stop_sign, y_stop_sign, transform=transform)
stop_sign_loader = DataLoader(stop_sign_dataset, batch_size=stop_sign_dataset.__len__(), shuffle=False)

print(X_stop_sign.shape)

# ## 2.1 Untargeted attack using Fast Gradient Sign Method (FGSM)


def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def plot_images(X, y, yp, M, N):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M*1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1-X[i*N+j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("Pred: {}".format(yp[i*N+j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i*N+j].max(dim=0)[1] == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()


# TODO: Interesting hack!
for X, y in stop_sign_loader:
    X, y = X.to(device), y.to(device)
    break

print(X.shape)
print(y.shape)


# Illustrate original predictions
yp = model_cnn(X)
plot_images(X, y, yp, 6, 6)


# Illustrate attacked images
delta = fgsm(model_cnn, X, y, 0.01)
yp = model_cnn(X + delta)
plot_images(X+delta, y, yp, 6, 6)


def epoch_adversarial(model, loader, attack, *args):
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp, y)

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


print("Epoch error :", epoch_adversarial(model_cnn, stop_sign_loader, fgsm, 0.01)[0])

# ## 2.2 Untargeted attack using Projected Gradient Descent

# The (normalized) steepest descent


def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


# Illustrate attacked images
delta = pgd_linf(model_cnn, X, y, epsilon=0.01, alpha=1e-2, num_iter=40)
yp = model_cnn(X + delta)
plot_images(X+delta, y, yp, 6, 6)


print("Epoch error :", epoch_adversarial(model_cnn, stop_sign_loader, pgd_linf, 0.01, 1e-2, 40)[0])

# ## 2.3 Targeted attack using Projected Gradient Descent ver 1

# Switch to the Speed Limit 50 km/h signs
for X, y in test_loader:
    X, y = X.to(device), y.to(device)
    break

print(X.shape)
print(y.shape)


def pgd_linf_targ(model, X, y, epsilon, alpha, num_iter, y_targ):
    """ Construct targeted adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        yp = model(X + delta)
        loss = (yp[:, y_targ] - yp.gather(1, y[:, None])[:, 0]).sum()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


# Illustrate original predictions
yp = model_cnn(X)
plot_images(X, y, yp, 6, 6)


delta = pgd_linf_targ(model_cnn, X, y, epsilon=0.03, alpha=1e-2, num_iter=40, y_targ=speed_limit_50_class_id)
yp = model_cnn(X + delta)
plot_images(X+delta, y, yp, 6, 6)


# An example of a successful attack
i = 12-1
image = tensor_image_to_image(X[i])
show_image(image, 'Original: ' + str(y[i].numpy()))
image = tensor_image_to_image(X[i] + delta[i])
show_image(image, 'Attacked: ' + str(yp[i].max(dim=0)[1].numpy()))

epoch_error = epoch_adversarial(
    model_cnn,
    test_loader,
    pgd_linf_targ2, 0.03, 1e-2, 40, speed_limit_50_class_id)[0]

print("Epoch error :", epoch_error)

print(f'Accuracy = {100*(1-epoch_error):.2f}%')

# ## 2.4 Targeted attack using Projected Gradient Descent ver 2


def pgd_linf_targ2(model, X, y, epsilon, alpha, num_iter, y_targ):
    """ Construct targeted adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        yp = model(X + delta)
        loss = 2*yp[:, y_targ].sum() - yp.sum()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


delta = pgd_linf_targ2(model_cnn, X, y, epsilon=0.03, alpha=1e-2, num_iter=40, y_targ=speed_limit_50_class_id)
yp = model_cnn(X + delta)
plot_images(X+delta, y, yp, 6, 6)


# Number 11 is not a success anymore, but...
# An example of a successful attack
i = 5*6+1-1
image = tensor_image_to_image(X[i])
show_image(image, 'Original: ' + str(y[i].numpy()))
image = tensor_image_to_image(X[i] + delta[i])
show_image(image, 'Attacked: ' + str(yp[i].max(dim=0)[1].numpy()))

epoch_error = epoch_adversarial(
    model_cnn,
    test_loader,
    pgd_linf_targ2, 0.03, 1e-2, 40, speed_limit_50_class_id)[0]

print("Epoch error :", epoch_error)

print(f'Accuracy = {100*(1-epoch_error):.2f}%')

# The End
