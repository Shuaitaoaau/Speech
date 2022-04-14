#!/usr/bin/env python
# coding=utf-8

import torch
import soundfile as sf
import librosa
import numpy as np
import gc

from time import sleep
from tqdm import tqdm

class Frame_Data():
    def __init__(self, para, noisy_file, clean_file, noise_file):
        self.noisy_file = noisy_file
        self.clean_file = clean_file
        self.noise_file = noise_file
        self.n_expand = para.n_expand


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
        return mag.T

    def get_frame(self, data, n):
        frames = data.unfold(0, 2 * n + 1, 1)
        frames = frames.transpose(1, 2)
        frames = frames.view([-1, (2 * n + 1) * frames.shape[-1]])
        return frames


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
            noisy_temp, fs1 = sf.read(noisy_file[i], dtype = 'float32')
            clean_temp, fs2 = sf.read(clean_file[i], dtype = 'float32')
            noise_temp, fs3 = sf.read(noise_file[i], dtype = 'float32')
            noisy_temp = noisy_temp.astype('float32')
            clean_temp = clean_temp.astype('float32')
            noise_temp = noise_temp.astype('float32')
            

            if i == 0:
                noisy_data = noisy_temp
                clean_data = clean_temp
                noise_data = noise_temp
            else:
                noisy_data = np.append(noisy_data, noisy_temp)
                clean_data = np.append(clean_data, clean_temp)
                noise_data = np.append(noise_data, noise_temp)
                
            sleep(0.01)
        sleep(0.5)
            

        del noisy_temp, clean_temp, noise_temp
        gc.collect()



        #STFT
        noisy_mag = self.STFT(noisy_data)
        clean_mag = self.STFT(clean_data)
        noise_mag = self.STFT(noise_data)

        del noisy_data, clean_data, noise_data
        gc.collect()

        #caculate mask
        mask = (clean_mag ** 2 / (clean_mag ** 2 + noise_mag ** 2)) ** 0.5

        del clean_mag, noise_mag
        gc.collect()

        #torch type
        noisy_LPS = torch.from_numpy(np.log(noisy_mag ** 2))
        mask = torch.from_numpy(mask)

        del noisy_mag
        gc.collect()
        
        #spelice
        noisy_LPS = self.get_frame(noisy_LPS, self.n_expand)
        
        maks = mask[self.n_expand : -self.n_expand, :]

        #normalize
        data_mean = noisy_LPS.mean()
        data_std = noisy_LPS.std()

        torch.save(data_mean, 'data_mean.pt')
        torch.save(data_std, 'data_std.pt')

        noisy_LPS = (noisy_LPS - data_mean) / data_std



        return noisy_LPS, mask




