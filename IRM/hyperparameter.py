#!/usr/bin/env python
# coding=utf-8

import torch

class hyperparameter():
    def __init__(self):
        self.n_expand = 3
        self.stft_nfft = 256
        
        self.dim_embeding = 1024
        


        self.dim_in = int((self.stft_nfft / 2 + 1) * (2 * self.n_expand + 1))
        self.dim_out = int(self.stft_nfft / 2 + 1)


        self.train_path = '/home/ts/Speech/SPP/DNSdataset/train.csv'
        self.save_path = '/home/ts/Speech/SPP/model_save'


        #For evaluate
        self.data_mean_path = '/home/ts/Speech/SPP/data_mean.pt'
        self.data_std_path = '/home/ts/Speech/SPP/data_std.pt'
        self.test_file_path = '/home/ts/Speech/SPP/DNSdataset/test.csv'
        self.enh_file_path = '/home/ts/Speech/SPP/Enhance'
