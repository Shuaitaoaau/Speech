import torch
from hyperparameter import hyperparameter
import os
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pysepm
import librosa.display
import gc

from train import GRU_REG
from numpy.linalg import norm
from pystoi import stoi
from pesq import pesq
from objective import SI_SDR

def feature_STFT(data, para):
    spec = librosa.stft(data,
                        n_fft = 256,
                        win_length = 256,
                        hop_length = 128,
                        window = 'hamming')
    mag = np.abs(spec)
    phase = np.angle(spec)
    del spec
    gc.collect()

    return mag, phase


def feature_contex(noisy_frames, expand):
    noisy_frames = noisy_frames.unfold(0, 2 * expand + 1, 1)
    noisy_frames = noisy_frames.transpose(1, 2)
    noisy_frames = noisy_frames.view([-1, (2 * expand + 1) * noisy_frames.shape[-1]])
    return noisy_frames



def eval_file_IRM(wav_file, model, para):
    
    noisy_data, fs = sf.read(wav_file,dtype = 'float32')
    noisy_data = noisy_data.astype('float32')

    noisy_mag, noisy_phase = feature_STFT(noisy_data, para)
    

    noisy_mag = torch.from_numpy(noisy_mag)
    noisy_mag1 = noisy_mag.unsqueeze(-1)
    noisy_mag1 = torch.transpose(noisy_mag1, 1, 0) #(seq_len, batch_size, input_size)
    


    model.eval()
    with torch.no_grad():
        enh_mask = model(x = noisy_mag1)

    enh_mask = enh_mask.numpy()
    noisy_mag = noisy_mag.numpy()
    print(enh_mask)

    enh_mag = noisy_mag * (1 - enh_mask)
    
    

    enh_spec = enh_mag * np.exp(1j * noisy_phase)
    
    # istft
    enh_wav = librosa.istft(enh_spec, hop_length = 128, win_length = 256)
    return enh_wav



def spectrogram_show(clean, noisy, enh):
    fs = 16000

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.specgram(clean, NFFT = 256, Fs = fs)
    plt.xlabel('clean spctrogram')
    plt.title('clean speech')
    plt.subplot(3, 1, 2)
    plt.specgram(noisy, NFFT = 256, Fs = fs)
    plt.xlabel('noisy spectrogram')
    plt.title('noisy speech')
    plt.subplot(3, 1, 3)
    plt.specgram(enh, NFFT = 256, Fs = fs)
    plt.xlabel('enhance spectrogram')
    plt.title('enhanced speech')
    plt.show()


def wav_show(clean, noisy, enh, sr):

    plt.figure(2)
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(clean, sr=sr)
    plt.title('clean speech')
    
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(noisy, sr=sr)       
    plt.title('noisy speech')

    plt.subplot(3, 1, 3)
    librosa.display.waveshow(enh, sr=sr)
    plt.title('enhanced speech')
    plt.show()


   
    
if __name__ == "__main__":
    
    para = hyperparameter()
    
    model_name = "/home/ts/SPEECH/GRU/model_save/model_49_0.0105.pth"
    m_model = torch.load(model_name,map_location = torch.device('cpu'))
    

    test_file = np.loadtxt(para.test_file_path, dtype = 'str')

    noisy_file = test_file[:, 0].tolist()
    clean_file = test_file[:, 1].tolist()

    file_len = len(noisy_file)
    
    for i in range(file_len):
        clean_data, fs1 = sf.read(clean_file[i], dtype = 'float32')
        noisy_data, fs2 = sf.read(noisy_file[i], dtype = 'float32')
        noisy_data = noisy_data.astype('float32')
        clean_data = clean_data.astype('float32')
        
        #enhance speech
        enh_data = eval_file_IRM(noisy_file[i], m_model, para)
        


        #evaluate
        len3 = len(enh_data)
        clean_data = clean_data[0 : len3]
        noisy_data = noisy_data[0 : len3]


        print('enh_stoi = %f'%stoi(clean_data, enh_data, fs2))
        print('noisy_stoi = %f'%stoi(clean_data, noisy_data, fs2))

        si_sdr1 = SI_SDR(enh_data, clean_data)
        si_sdr2 = SI_SDR(noisy_data, clean_data)

        print('enh si_sdr = %f'%si_sdr1)
        print('noisy si_sdr = %f'%si_sdr2)

        segSNR = pysepm.SNRseg(clean_data, enh_data, fs2)
        pesq_score = pesq(fs1, clean_data, enh_data, 'wb')

        segSNR1 = pysepm.SNRseg(clean_data, noisy_data, fs1)
        pesq_score1 = pesq(fs1, clean_data, noisy_data, 'wb')


        print('enh segSNR = %f'%segSNR)
        print('enh pesq = %f'%pesq_score)

        print('noisy segSNR = %f'%segSNR1)
        print('noisy pesq = %f'%pesq_score1)

        print('============================')




        #write the enhance file
        enh_file_name = str(i) + 'enhance' + '.wav'
        enh_file_name = os.path.join(para.enh_file_path, enh_file_name)
        sf.write(enh_file_name, enh_data, fs1)

        
        #spectrogram
        spectrogram_show(clean_data, noisy_data, enh_data)


        #waveshow
        wav_show(clean_data, noisy_data, enh_data, fs2)

                
                
                
                
               
    
 
    
    
