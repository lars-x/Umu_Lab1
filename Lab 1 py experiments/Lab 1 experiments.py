import numpy as np
import matplotlib.pyplot as plt

import os
import urllib.request
import zipfile
import pickle

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
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.show()


def show_images(X, y, indexes, cols):
    n = len(indexes)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, squeeze=False)
    i = 0
    for r in range(rows):
        for c in range(cols):
            axs[r, c].axis('off')
            if i >= n:
                continue
            index = indexes[i]
            axs[r, c].imshow(X[index])
            axs[r, c].set_title(f'Class = {y[index]}')
            i = i + 1

    fig.tight_layout()
    fig.set_figheight(rows*4)
    fig.set_figwidth(8)
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
show_images(X_train, y_train, indexes, cols=2)

print('Show some random Stop signs')
indexes_random, indexes_total = get_n_random_indices_for_class(y_train, 10, 14)
show_images(X_train, y_train, indexes_random, cols=10)

print('Show traffic signs distibution')
classes, counts = np.unique(y_train, return_counts=True)
plot_image_distribution(classes, counts, 'Classes', '# Training Examples', 0.7, 'blue')
