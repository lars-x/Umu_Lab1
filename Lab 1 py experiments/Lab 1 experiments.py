import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import urllib.request
import zipfile
import os

data_dir = './data/'
training_file = data_dir + 'train.p'
validation_file = data_dir + 'valid.p'
testing_file = data_dir + 'test.p'
zip_file = data_dir + 'traffic-signs-data.zip'

if not (os.path.exists(training_file) and os.path.exists(validation_file) and os.path.exists(testing_file)):
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

print("Number of training examples   =", X_train.shape[0])
print("Number of validation examples =", X_valid.shape[0])
print("Number of testing examples    =", X_test.shape[0])
print("Total examples                =", X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
# What's the shape of an traffic sign image?
print("Image data shape =", X_train.shape[1:])
# How many unique classes/labels there are in the dataset.
print("Number of classes =", len(np.unique(y_train)))

i = 1
image = X_train[i]
title = f'Image {i}'


plt.imshow(image)
plt.axis('off')
plt.title(title)
plt.show()

def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    fig = plt.figure(figsize=(2, 2))

    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        a.grid(False)
        a.axis('off')
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, cmap='gray')
        a.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def select_random_images_by_classes(features, labels, n_features):

    indexes = []
    _classes = np.unique(labels)

    while len(indexes) < len(_classes):

        index = random.randint(0, n_features-1)
        _class = labels[index]

        for i in range(0, len(_classes)):

            if _class == _classes[i]:
                _classes[i] = -1
                indexes.append(index)
                break

    images = []
    titles = []

    for i in range(0, len(indexes)):
        images.append(features[indexes[i]])
        titles.append("class " + str(labels[indexes[i]]))

    show_images(images, titles=titles)


# Data exploration visualization code goes here.
# Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.
%matplotlib inline

select_random_images_by_classes(X_train, y_train, n_train)


def plot_distribution_chart(x, y, xlabel, ylabel, width, color):

    plt.figure(figsize=(15, 7))
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.bar(x, y, width, color=color)
    plt.show()


_classes, counts = np.unique(y_train, return_counts=True)

plot_distribution_chart(_classes, counts, 'Classes', '# Training Examples', 0.7, 'blue')

# 3. Convert images to grayscale ---------------------------------

X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)

X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)

X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)

# check grayscale images
select_random_images_by_classes(X_train_gray.squeeze(), y_train, n_train)

# 4. Mean Substraction

X_train_gray -= np.mean(X_train_gray)

X_test_gray -= np.mean(X_test_gray)

X_train = X_train_gray

X_test = X_test_gray

select_random_images_by_classes(X_train_gray.squeeze(), y_train, n_train)
