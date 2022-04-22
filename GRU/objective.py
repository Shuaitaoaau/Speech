#!/usr/bin/env python
# coding=utf-8

import numpy as np

def SI_SDR(estimate, reference):
    """Scale-Invariant Signal to Distortion Ratio (SI-SDR)
    :param estimate:估计的信号
    :param reference:参考信号
    """
    eps = np.finfo(np.float).eps
    alpha = np.dot(estimate.T, reference) / (np.dot(estimate.T, estimate) + eps)
    #print(alpha)

    molecular = ((alpha * reference) ** 2).sum()  # 分子
    denominator = ((alpha * reference - estimate) ** 2).sum()  # 分母

    return 10 * np.log10((molecular) / (denominator+eps))
