#!/usr/bin/env python
# coding: utf8
# Author: Tong ZHAO

import numpy as np

def sim2_log(mat, theta = None):
    """From SIM(2) to sim(2) by applying logarithmic map

    Params:
        mat    (np.array): transformation matrix of size 3 x 3
        theta  (float)   : if theta is known, it will not be re-calculated
    Returns:
        log_mat (np.array): an element in sim(2) of size 4 x 1
    """
    
    log_mat = np.zeros((4,))
    
    if theta is  None:
        theta = np.arccos(mat[0, 0])
        sign = 1 if mat[1, 0] > 0 else -1
        theta = (theta * sign) % (2 * np.pi)
        
    log_mat[2] = theta
    
    param_a = theta * mat[1, 0] / np.clip(2 * (1 - mat[0, 0]), 1e-6, None)
    param_b = theta / 2
    
    log_mat[0] = param_a * mat[0, 2] + param_b * mat[1, 2]
    log_mat[1] = param_a * mat[1, 2] - param_b * mat[0, 2]

    log_mat[3] = np.log(mat[2, 2])
    
    return log_mat


def sim2_exp(log_mat):
    """From sim(2) to SIM(2) by applying exponential map

    Params:
        log_mat (np.array): an element in sim(2) of size 4 x 1
    Returns:
        mat (np.array): an element in SIM(2) of size 3 x 3
    """
    
    
    mat = np.zeros((3, 3))
    
    theta = log_mat[2]
    mat[0, 0] = np.cos(theta)
    mat[0, 1] = -np.sin(theta)
    mat[1, 0] = np.sin(theta)
    mat[1, 1] = np.cos(theta)
    
    mat[2, 2] = np.exp(log_mat[3])
    
    param_a = mat[1, 0] / theta
    param_b = (1 - mat[1, 1]) / theta
    mat[0, 2] = param_a * log_mat[0] - param_b * log_mat[1]
    mat[1, 2] = param_b * log_mat[0] + param_a * log_mat[1]

    return mat


def transform_embedding(lT):
    """Map a list of elements in SIM(2) to sim(2)

    Params:
        lT (list[np.array]): elements in SIM(2)
    Returns:
        lT_log (list[np.array]): corresponding elements in sim(2)
    """

    lT_log = np.zeros((len(lT), 4))

    for i in range(len(lT)):
        lT_log[i] = sim2_log(lT[i])

    return lT_log