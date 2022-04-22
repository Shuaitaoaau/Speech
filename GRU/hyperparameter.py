#!/usr/bin/env python
# coding=utf-8


import torch

class hyperparameter():
    def __init__(self):

        self.lr = 1e-4



        self.train_path = '/home/ts/DNSdataset/train.csv'
        self.save_path = '/home/ts/SPEECH/GRU/model_save'


        self.input_size = 1
        self.hidden_size = 1
        self.num_layes = 1
        self.seq_len = 2501
        self.output_size = 1
        
        #evaluate
        self.test_file_path = '/home/ts/DNSdataset/test.csv' 
        self.enh_file_path = '/home/ts/SPEECH/GRU/Enhance'

