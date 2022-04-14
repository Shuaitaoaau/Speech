#!/usr/bin/env python
# coding=utf-8


import torch
import os
import librosa
import soundfile as sf
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from hyperparameter import hyperparameter
from feature_extract import Frame_Data


class DNSDataset(Dataset):
    def __init__(self, noisy_frame, clean_frame):
        self.noisy_frame = noisy_frame
        self.clean_frame = clean_frame
        self.len = len(self.noisy_frame)



    def __getitem__(self, index):
        return self.noisy_frame[index, :], self.clean_frame[index, :]




    def __len__(self):
        
        return self.len



class Sp_En_Model(torch.nn.Module):
    def __init__(self, para):
        super(Sp_En_Model, self).__init__()
        self.dim_in = para.dim_in
        self.dim_out = para.dim_out
        self.dim_embeding = para.dim_embeding
        self.BNlayer = nn.BatchNorm1d(self.dim_out)

        self.linear1 = torch.nn.Linear(self.dim_in, self.dim_embeding)
        self.linear2 = torch.nn.Linear(self.dim_embeding, self.dim_embeding)
        self.linear3 = torch.nn.Linear(self.dim_embeding, self.dim_embeding)
        self.linear4 = torch.nn.Linear(self.dim_embeding, self.dim_out)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        #nn.Dropout(0.1)
        x = self.sigmoid(self.linear2(x))
        #nn.Dropout(0.1)
        x = self.sigmoid(self.linear3(x))
        #nn.Dropout(0.1)
        x = self.sigmoid(self.linear4(x))
        return x





if __name__ == '__main__':
    
    para = hyperparameter()
    files = np.loadtxt(para.train_path, dtype = 'str')
    noisy_files = files[:, 0].tolist()
    clean_files = files[:, 1].tolist()
    noise_files = files[:, 2].tolist()
    

    len1 = len(noisy_files)
    


    model = Sp_En_Model(para)
    model.train()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-4)

    cnt = 0
    loss_sum = 0
    temp = 0
    x_temp = []
    y_temp = []


    Data = Frame_Data(para, noisy_files, clean_files, noise_files)
    noisy_frame, IRM_frame = Data.feature_extract()


    dataset = DNSDataset(noisy_frame, IRM_frame)
    train_loader = DataLoader(dataset = dataset,
                            batch_size = 128,
                            shuffle = True,
                            num_workers = 4)
    for epoch in range(7):
        print('epoch = %d'%epoch)
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data

            model.zero_grad()

            y_pred = model(inputs)

            loss = criterion(y_pred, labels)

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

                












            




            




        


        






