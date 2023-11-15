"""##Importing the libraries"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from torch.autograd import Variable


nb_aps = 4
training_set = pd.read_csv('New_SET.csv',usecols=np.r_[0:nb_aps])
training_set = np.array(training_set, dtype = 'int')


nb_fps =  training_set[:,0].size


def convert(data):
  new_data = []
  for id_fps in range(0, nb_fps):
    new_data.append(list(training_set[id_fps]))
  return new_data
training_set = convert(training_set)
#test_set = convert(test_set)
training_set = np.array(training_set,float)


training_set = torch.FloatTensor(training_set)
#test_set = torch.FloatTensor(test_set)


class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_aps, 2)
        self.fc2 = nn.Linear(2, 1)
        self.fc3 = nn.Linear(1, 2)
        self.fc4 = nn.Linear(2, nb_aps)
        self.p = 0.5
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = nn.functional.dropout(x, p=self.p, training=True)
        x = self.activation(self.fc2(x))
        x = nn.functional.dropout(x, p=self.p, training=True)
        x = self.activation(self.fc3(x))
        x = nn.functional.dropout(x, p=self.p, training=True)
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

"""## Training the SAE"""

nb_epoch = 25
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_fps in range(nb_fps):
    input = Variable(training_set[id_fps]).unsqueeze(0)
    target = input.clone()
    output = sae(input)
    target.require_grad = False
    loss = criterion(output, target)
    mean_corrector = nb_aps/float(4.0 + 1e-10)
    loss.backward()
    train_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
    optimizer.step()
  print('epoch: '+str(epoch)+' loss: '+ str(train_loss/s))
  
with open('auto_102', 'wb') as config_dictionary_file:
 
  # Step 3
  pickle.dump(sae, config_dictionary_file)
  
with open('auto_102', 'rb') as config_dictionary_file:
 
    # Step 3
    obj_r = pickle.load(config_dictionary_file)
 
