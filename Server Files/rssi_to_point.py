# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 20:37:25 2021

@author: HP
"""
from keras.models import load_model
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import socket
import codecs
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
tf.__version__

nb_aps = 4

class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_aps, 2)
        self.fc2 = nn.Linear(2, 1)
        self.fc3 = nn.Linear(1, 2)
        self.fc4 = nn.Linear(2, nb_aps)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
with open('auto_102', 'rb') as config_dictionary_file:
 
    obj_r = pickle.load(config_dictionary_file)
    
with open('mac_dictions', 'rb') as config_dictionary_file:
 
    mac_addr = pickle.load(config_dictionary_file)    
 
model = load_model('ANN_Project_NEW.h5')
sc = joblib.load('ann_project_scalar_NEW.gz')

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
print("Waiting....")
while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            while True:
                data = conn.recv(10000)
                ret = codecs.decode(data, 'UTF-8')
                print(ret)
                ltr = ret.split('*')
                ltr.pop(len(ltr) - 1)
                ap_diction = {}
                for i in range((int)(len(ltr)/2)):
                	ap_diction[ltr[2*i]] = ltr[2*i+1]
                ap_list = []
                for i in range(nb_aps):
                    ap_list.append(0)
                    
                for key in mac_addr:
                    if key in ap_diction:
                        ap_list[(int)(mac_addr[key])] = ap_diction[key]
                        
                rssi_pred = ap_list
                ap_list = np.array(ap_list,dtype = 'int')
                ap_list = torch.FloatTensor(ap_list)  
                output_try = obj_r(Variable(ap_list)).unsqueeze(0)
                output_try = output_try[0].detach().cpu().numpy()
                
                for i in range(len(rssi_pred)):
                    if rssi_pred[i] == 0:
                        rssi_pred[i] = (int)(output_try[i])
                    else:
                        rssi_pred[i] = (int)(rssi_pred[i])
                
                
                output = model.predict(sc.transform([rssi_pred]))
                
                output = output[0]
                result = 0
                inx = 0
                i = 0
                for x in output:
                    if x > result:
                        result = x
                        inx = i
                    i = i + 1
                print(inx)
                if not data:
                    break
                conn.sendall(bytes(str(inx), 'utf-8'))
                break
         


#ret = "1c:5f:2b:da:78:ec*-36*a6:ae:12:0e:37:ff*-38*1e:96:e6:3d:e2:df*-42*1e:4d:70:af:f8:9d*-71*"
