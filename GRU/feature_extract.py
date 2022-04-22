#!/usr/bin/env python
# coding=utf-8


import torch
import soundfile as sf
import librosa
import numpy as np
import gc

from hyperparameter import hyperparameter
from time import sleep
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class Frame_Data():
    def __init__(self, para, noisy_file, clean_file, noise_file):
        self.noisy_file = noisy_file
        self.clean_file = clean_file
        self.noise_file = noise_file


    def STFT(self, data):
        spec = librosa.stft(data,
                            n_fft = 256,
                            hop_length = 128,
                            win_length = 256,
                            window = 'hamming')
        mag = np.abs(spec)
        del spec
        gc.collect()
        #LPS = np.log(mag ** 2 + 1e-5)
        return mag


    def Norm_Func(self, data, data_mean, data_std):
        len1 = data.shape[1]

        for i in range(len1):
            data[:, i] = (data[:, i] - data_mean[i]) / data_std[i]
        return data


    def feature_extract(self):
        noisy_file = self.noisy_file
        clean_file = self.clean_file
        noise_file = self.noise_file


        file_len = len(noisy_file)

        print("======TASK 1: Load Speech Data======")

        for i in tqdm(range(file_len)):

            #read file
            noisy_data, fs1 = sf.read(noisy_file[i], dtype = 'float32')
            clean_data, fs2 = sf.read(clean_file[i], dtype = 'float32')
            noise_data, fs3 = sf.read(noise_file[i], dtype = 'float32')
            noisy_data = noisy_data.astype('float32')
            clean_data = clean_data.astype('float32')
            noise_data = noise_data.astype('float32')


            #STFT
            noisy_temp = self.STFT(noisy_data)
            clean_temp = self.STFT(clean_data)
            noise_temp = self.STFT(noise_data)



            if i == 0:
                noisy_frames = noisy_temp
                clean_frames = clean_temp
                noise_frames = noise_temp
            else:
                noisy_frames = np.vstack((noisy_frames, noisy_temp))
                clean_frames = np.vstack((clean_frames, clean_temp))
                noise_frames = np.vstack((noise_frames, clean_temp))

            sleep(0.01)
        sleep(0.5)

        del noisy_data, clean_data, noise_data, noisy_temp, clean_temp, noise_temp
        gc.collect()
        
        data_mean = noisy_frames.mean()
        data_std = noisy_frames.std()

        torch.save(data_mean, 'data_mean.pt')
        torch.save(data_std, 'data_std.pt')

        noisy_frames = (noisy_frames - data_mean) / data_std


        #caculate mask
        mask = (clean_frames ** 2 / (clean_frames ** 2 + noise_frames ** 2 + 1e-7)) ** 0.5


        return noisy_frames, mask


class DNSDataset(Dataset):
    def __init__(self, noisy_frame, clean_frame):
        self.noisy_frame = noisy_frame
        self.clean_frame = clean_frame
        self.len = len(self.noisy_frame)


    def __getitem__(self, index):
        return self.noisy_frame[index, :], self.clean_frame[index, :]


    def __len__(self):
        
        return self.len




if __name__ == "__main__":

    
    para = hyperparameter()

    files = np.loadtxt(para.train_path, dtype = 'str')
    noisy_files = files[:, 0].tolist()
    clean_files = files[:, 1].tolist()
    noise_files = files[:, 2].tolist()
    

    Data = Frame_Data(para, noisy_files, clean_files, noise_files)
    noisy_frame, IRM_frame = Data.feature_extract()


    dataset = DNSDataset(noisy_frame, IRM_frame)
    train_loader = DataLoader(dataset = dataset,
                            batch_size = 129,
                            shuffle = True,
                            num_workers = 4)

    for j, data in enumerate(train_loader, 0):
        inputs, labels = data

        #inputs: (batch_size, seq_len, input_size) => (seq_len, batch_size, input_size)

        print(inputs.shape)

        inputs = torch.transpose(inputs, 1, 0) #(seq_len, batch_size, input_size)
        print(labels.shape)
