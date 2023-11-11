from ImageUtils import parse_record
from DataReader import load_data, train_vaild_split
from Model import Cifar
from torchsummary import summary
import torch
import numpy as np
import os
import argparse

def configure():
    parser = argparse.ArgumentParser()
    ### YOUR CODE HERE
    parser.add_argument("--resnet_version", type=int, default=1, help="the version of ResNet")
    parser.add_argument("--resnet_size", type=int, default=18, 
                        help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10, 
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--first_num_filters", type=int, default=16, help='number of classes')
    parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default='model_v1', help='model directory')
    parser.add_argument("--learning_rate", type=float, default=0.1, help='learning rate')
    parser.add_argument("--model_number", type=int, default=1, help='model number')
    ### YOUR CODE HERE
    return parser.parse_args()

def main(config):
    print("--- Preparing Data ---")

    ### YOUR CODE HERE
    data_dir = "/gpfs/scratch/kgayadhankar/Resnet_v1/cifar-10-batches-py"
    ### YOUR CODE HERE

    x_train, y_train, x_test, y_test = load_data(data_dir)
    print("cifar_batch")
    x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train)
    shuffle_index = np.random.permutation(x_train_new.shape[0])[:5000]
    x_train_new = x_train_new[shuffle_index]
    y_train_new = y_train_new[shuffle_index]
    #model = Cifar(config).cuda()
    """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)"""
    
    #summary(model, (3, 32, 32))

    ### YOUR CODE HERE
    batch_sizes = [32,128,512]
    learning_rates = [0.1,0.01]
    weight_decays = [0.002,0.0002]
    # First step: use the train_new set and the valid set to choose hyperparameters.
    for i in batch_sizes:
        for j in learning_rates:
            for k in weight_decays:
                config.batch_size = i
                config.learning_rate = j
                config.weight_decay = k
                print("The hyperparameters for model",config.model_number,"used are: Batch Size",config.batch_size,"Learning Rate:",config.learning_rate,"Weight Decay:",config.weight_decay)
                model = Cifar(config).cuda()
                model.train(x_train_new, y_train_new, 200)
                model.test_or_validate(x_valid, y_valid, [160, 170, 180, 190, 200])
                config.model_number += 1
    #model.train(x_train_new, y_train_new, 200)
    #model.test_or_validate(x_valid, y_valid, [160, 170, 180, 190, 200])
    # Second step: with hyperparameters determined in the first run, re-train
    # your model on the original train set.
    # model.train(x_train, y_train, 10)

    # Third step: after re-training, test your model on the test set.
    # Report testing accuracy in your hard-copy report.
    # model.test_or_validate(x_test, y_test, [10])
    ### END CODE HERE

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)