import numpy as np
import os

from pathlib import Path

class DataLoader():
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.x_train, self.y_train = self.train_load_data()
        self.x_test, self.y_test = self.test_load_data() 
        
        print(self.data_path + "/x_train.npy : ", self.x_train.shape)
        print(self.data_path + "/y_train.npy : ", self.y_train.shape)
        print(self.data_path + "/x_test.npy  : ", self.x_test.shape)
        print(self.data_path + "/y_test.npy  : ", self.y_test.shape)  
        
    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test
            
    def train_load_data(self):
        x_train = np.load(self.data_path + '/x_train.npy')
        y_train = np.load(self.data_path + '/y_train.npy')
        return (x_train, y_train)

    def test_load_data(self):
        x_test = np.load(self.data_path + '/x_test.npy')
        y_test = np.load(self.data_path + '/y_test.npy')
        return (x_test, y_test) 