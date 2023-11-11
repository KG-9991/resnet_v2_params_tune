import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    train_data = []
    y_train = []

    for i in range(1, 6):  # CIFAR-10 is split into 5 batches
        with open(f'{data_dir}/data_batch_{i}', 'rb') as fo:
            batch_data = pickle.load(fo, encoding='bytes')
        train_data.append(batch_data[b'data'])
        y_train += batch_data[b'labels']

    x_train = np.concatenate(train_data, axis=0)

    # Load the test data
    with open(f'{data_dir}/test_batch', 'rb') as fo:
            test_data = pickle.load(fo, encoding='bytes')
    x_test = np.array(test_data[b'data'])
    y_test = np.array(test_data[b'labels'])

    return x_train, np.array(y_train), x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid
