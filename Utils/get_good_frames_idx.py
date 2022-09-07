# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:20:34 2021

@author: Fredrik Skärberg
"""


import cv2, numpy as np


def frame_check(video, index=1):
    """
    Computes a metric: mean/std to quantify how "normal" a frame is.
    """
    
    video.set(1, index); # Where index is the frame you want
    _, frame = video.read() # Read the frame
    image = frame[:, :, 0]  
        
    mean = np.mean(image)
    std = np.std(image)
    
    return mean/std

def good_frames(video, good_frames_idx = np.arange(300), cap = 1000):
    """
    Remove frames that "stick out" from the rest. I.e that are different.
    """

    good_frames_idx = np.array(good_frames_idx, int)
    
    if cap == 0:
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of frames
    else:
        n_frames = cap
    
    #Compute metrics for frames.
    metrics = [frame_check(video, index=i) for i in range(n_frames)]
    
    metris_good_m = np.mean(np.array(metrics)[good_frames_idx])
    metris_good_std = np.std(np.array(metrics)[good_frames_idx])
    
    #These are "normal" frames. Not perfect, but removes outliers.
    good = [i for i, m in enumerate(metrics) if metris_good_m - 2*metris_good_std <= m <= metris_good_m +2*metris_good_std]  
    
    return np.array(good)

