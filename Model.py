import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr = self.config.learning_rate, momentum=0.9, weight_decay=self.config.weight_decay)
        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        learning_rate = self.config.learning_rate
        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            #print("Shapeeeeee",x_train.shape)
            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            if epoch % 60 == 0:
                learning_rate = learning_rate / 10
                self.optimizer.param_groups[0]['lr'] = learning_rate
            ### YOUR CODE HERE
            train_loss = 0
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                if i == num_batches - 1 and num_samples%num_batches != 0:
                    #print("yes last batch")
                    x_train_new = curr_x_train[self.config.batch_size*num_batches:]
                    y_train_new = curr_y_train[self.config.batch_size*num_batches:]
                    #print("batch_shape",x_train_new.shape)
                else:
                    x_train_new = curr_x_train[i*(self.config.batch_size):(i+1)*(self.config.batch_size)]
                    #print("batch shape",x_train_new.shape,flush=True)
                    y_train_new = curr_y_train[i*(self.config.batch_size):(i+1)*(self.config.batch_size)]
                x_train_pre = []
                for j in range(x_train_new.shape[0]):
                    x_train_pre.append(parse_record(x_train_new[j],True)) 
                x_train_pre = torch.tensor(np.array(x_train_pre), dtype=torch.float32)
                x_train_pre = x_train_pre.cuda()
                #print("Batch number:",i+1)
                outputs = self.network(x_train_pre)
                y_train_new = torch.tensor(y_train_new)
                y_train_new = y_train_new.cuda()
                loss = self.loss(outputs,y_train_new)             
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * self.config.batch_size
                print("Batch {:d}/{:d} Loss {:.6f}".format(i, num_batches, loss.item()),end='\r',flush=True)
                #print("Batch {:d}/{:d} Loss {:.6f}".format(i, num_batches, loss), end='\r', flush=True)
            avrg_loss = train_loss/num_samples
            print("Average training loss for the epoch",epoch,"is",avrg_loss)
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration),flush=True)

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        """x = x.reshape(-1, 3, 32, 32)
        x = torch.tensor(x, dtype=torch.float32)
        x = x.cuda()"""
        x_test_pre = []
        for i in x:
                x_test_pre.append(parse_record(i,False))
        x_test_pre = torch.tensor(x_test_pre)
        x_test_pre = x_test_pre.cuda()
        # Now you can pass x_tensor to the network
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir+'v2-%d'%(self.config.model_number), 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)
            ### YOUR CODE HERE
            preds = []
            #x_test_pre = []
            for i in tqdm(range(x.shape[0])):
                with torch.no_grad():
                    outputs = self.network(x[i].unsqueeze(0))
                _, predicted = torch.max(outputs, 1)
                preds.append(predicted.item())
            
            ### END CODE HERE

            y = torch.tensor(y)
            y = y.cuda()
            preds = torch.tensor(preds)
            preds = preds.cuda()
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir+'v2-%d'%(self.config.model_number), 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

