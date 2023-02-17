# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:26:19 2021

@author: Fredrik Skärberg
"""
import numpy as np

def get(idx, start_frame, vid_shift, frame_disp_vid, max_frames, method = 'old', every = 9, index = []):
    "Function for retrieving which indeces to extract, contains a few methods."
    
    if method == 'old':
        #Which frames to take out.
        r1 = np.arange(start_frame + 1, len(idx), vid_shift)
        r2 = np.arange(start_frame + 2, len(idx), vid_shift)
        if frame_disp_vid > 2:
            r3 = np.arange(start_frame + 3, len(idx), vid_shift)
        else: r3 = []

        r1 = idx[np.array(r1, dtype = int)]
        r2 = idx[np.array(r2, dtype = int)]
        r3 = idx[np.array(r3, dtype = int)]
        
        #Concatenate the frames
        input_mp = np.array(np.round(
            np.sort(np.concatenate((r1, r2, r3), axis = 0))), dtype = int)
          
    # All frames.            
    elif method == 'all':
        if max_frames < len(idx):
            input_mp = idx[start_frame:max_frames+start_frame]
        else:
            input_mp = idx
    
    elif method == 'pre2':
        r1 = np.arange(start_frame + 3, len(idx), vid_shift)
        r2 = np.arange(start_frame + 4, len(idx), vid_shift)
        input_mp = np.array(np.round(
                    np.sort(np.concatenate((r1, r2), axis = 0))), dtype = int)
        
    elif method == 'pre3':
        r1 = np.arange(start_frame + 3, len(idx), vid_shift)
        r2 = np.arange(start_frame + 4, len(idx), vid_shift)
        r3 = np.arange(start_frame + 5, len(idx), vid_shift)
        input_mp = np.array(np.round(
                    np.sort(np.concatenate((r1, r2, r3), axis = 0))), dtype = int)

    elif method == 'pre4':
        r1 = np.arange(start_frame + 3, len(idx), vid_shift)
        r2 = np.arange(start_frame + 4, len(idx), vid_shift)
        r3 = np.arange(start_frame + 5, len(idx), vid_shift)
        r4 = np.arange(start_frame + 6, len(idx), vid_shift)
        input_mp = np.array(np.round(
                    np.sort(np.concatenate((r1, r2, r3, r4), axis = 0))), dtype = int)

    elif method == 'pre5':
        r1 = np.arange(start_frame + 3, len(idx), vid_shift)
        r2 = np.arange(start_frame + 4, len(idx), vid_shift)
        r3 = np.arange(start_frame + 5, len(idx), vid_shift)
        r4 = np.arange(start_frame + 6, len(idx), vid_shift)
        r5 = np.arange(start_frame + 7, len(idx), vid_shift)
        input_mp = np.array(np.round(
                    np.sort(np.concatenate((r1, r2, r3, r4, r5), axis = 0))), dtype = int)

    elif method == 'prepost':
        #start frame is in this regime the first shift.
        shift_frames = np.arange(start_frame, len(idx), vid_shift)

        if start_frame - frame_disp_vid < 0:
            print("Error, negative start frame, fix not implemented yet")
 
        input_mp = []
        extra_disp = 2
        for shift in shift_frames:
            for k in np.arange(-frame_disp_vid, frame_disp_vid + 1):
                #Dont consider the shift frame
                if k < 0:
                    input_mp.append(shift-(np.abs(k) + extra_disp))
                if k>0:
                    input_mp.append(shift + k + extra_disp)

    elif method == 'every':
        frames_to_extract = np.arange(start_frame, len(idx), every)
        input_mp = idx[frames_to_extract]

    elif method == 'own_idx':
        input_mp = index

    ####Cap the number of frames
    if max_frames !=1:
        if max_frames < len(input_mp):
            input_mp = input_mp[:max_frames]   
    
    return np.array(input_mp, dtype = int)