#!/usr/bin/env python
# coding=utf-8

import torch
import matplotlib.pyplot as plt





if __name__ == "__main__":
    
    x_temp = torch.load('x_temp.mat')
    y_temp = torch.load('y_temp.mat')


    plt.plot(x_temp, y_temp)
    plt.xlabel('times / 100')
    plt.ylabel('ERROR')
    plt.title('Adam')
    plt.show()
