#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from feature_extract import Frame_Data 
from torch.utils.data import Dataset, DataLoader
from hyperparameter import hyperparameter



class DNSDataset(Dataset):
    def __init__(self, noisy_frame, clean_frame):
        self.noisy_frame = noisy_frame
        self.clean_frame = clean_frame
        self.len = len(self.noisy_frame)


    def __getitem__(self, index):
        return self.noisy_frame[index, :], self.clean_frame[index, :]


    def __len__(self):
        
        return self.len



class GRU_REG(nn.Module):
    def __init__(self, para):

        self.input_size = para.input_size
        self.hidden_size = para.hidden_size
        self.num_layers = para.num_layes
        self.seq_len = para.seq_len
        self.output_size = para.output_size

        
        super(GRU_REG, self).__init__()
        
        
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        
        for name, param in self.gru.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        
        self.reg = nn.Linear(self.seq_len, self.seq_len) # regression

        
    def forward(self, x):
        x, _ = self.gru(x) # (batch_size, seq_len, input_size)
        x = torch.transpose(x, 1, 0)
        b, s, h = x.shape
        x = x.reshape(b, s * h)
        x = self.reg(x)
        return x



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    para = hyperparameter()

    files = np.loadtxt(para.train_path, dtype = 'str')
    noisy_files = files[:, 0].tolist()
    clean_files = files[:, 1].tolist()
    noise_files = files[:, 2].tolist()
    

    model = GRU_REG(para)
    model.train()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = para.lr)

    cnt = 0
    loss_sum = 0
    temp = 0
    x_temp = []
    y_temp = []


    Data = Frame_Data(para, noisy_files, clean_files, noise_files)
    noisy_frame, IRM_frame = Data.feature_extract()


    dataset = DNSDataset(noisy_frame, IRM_frame)
    train_loader = DataLoader(dataset = dataset,
                            batch_size = 129,
                            shuffle = True,
                            num_workers = 4)

    for epoch in range(10):
        print('epoch = %d'%epoch)
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data

            #inputs: (batch_size, seq_len, input_size) => (seq_len, batch_size, input_size)

            inputs = inputs.unsqueeze(-1)

            inputs = torch.transpose(inputs, 1, 0) #(seq_len, batch_size, input_size)

            
            y_pred = model(inputs)


            loss = criterion(y_pred, labels)

            optimizer.zero_grad()

            loss_sum += loss.item()

            loss.backward()
            optimizer.step()


            cnt += 1
            if cnt % 100 == 0:
                loss_avrg = loss_sum / 100
                print('loss_avrg = %.4f'%loss_avrg)

                cnt = 0
                temp += 1
                x_temp.append(temp)
                y_temp.append(loss_avrg)
                loss_sum = 0

    #save the trained model
    model_name = os.path.join(para.save_path, 'model_%d_%.4f.pth'%(epoch, loss_avrg))
    torch.save(model, model_name)




    plt.plot(x_temp, y_temp)
    plt.xlabel('times / 100')
    plt.ylabel('ERROR')
    plt.title('Adam')
    plt.show()






