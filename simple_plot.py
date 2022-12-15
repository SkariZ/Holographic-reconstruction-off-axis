# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:23:54 2021

@author: Fredrik Skärberg
"""

import CONFIG

import matplotlib.pyplot as plt
import numpy as np, os, scipy, cv2
from scipy import ndimage

from Utils import phase_utils
from Utils import Utils_z
from Utils import image_utils

def downsample2d(inputArray, kernelSize):
    average_kernel = np.ones((kernelSize, kernelSize))

    blurred_array = scipy.signal.convolve2d(inputArray, average_kernel, mode='same')
    downsampled_array = blurred_array[::kernelSize,::kernelSize]
    return downsampled_array

#Save frame in native resolution. Can change colormap if necessary.
def save_frame(frame, folder, name, cmap = 'gray'):
    plt.imsave(f"{folder}/{name}.png", frame, cmap = cmap)
    
def subtract_data(data, frame_disp_vid, sign = 1):
    df = []
    for i in range(0, len(data)-frame_disp_vid):

        if sign==1:
            sub = data[i] - data[i+frame_disp_vid]
        if sign==0:
            sub = data[i+frame_disp_vid] - data[i]

        df.append(sub) #Append sub frame to df.

        if (i+1) % frame_disp_vid == 0 and sign == 0: sign = 1
        elif (i+1) % frame_disp_vid == 0 and sign == 1: sign = 0
        else: pass

    return df

def correctfield(field, n_iter = 5):
    """
    Correct field
    """
    f_new = field.copy()

    #Normalize with mean of absolute value.
    f_new = f_new / np.mean(np.abs(f_new))

    for _ in range(n_iter):
        f_new = f_new * np.exp(-1j * np.median(np.angle(f_new)))

    return f_new

def cropping_image(image, h, w, corner = 4):
    """
    Crops the image
    """
    
    hi, wi = image.shape[:2]
    if hi<=h or wi<=w:
        raise Exception("Cropping size larger than actual image size.")
    
    if wi == hi:#If we have a square image we can resize without loss of quality
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
    else: #Crop out the "corner"
        if corner == 1:
            image = image[:h, :w]#image[-h:, -w:] #image[:h, :w] # Important to keep check here which corner we look at.
        elif corner == 2:
            image = image[:h, -w:]
        elif corner == 3:
            image = image[-h:, :w]
        elif corner == 4:
            image = image[-h:, -w:]
    return image
    
def first_frame(video, height, width, index=1):
    """
    Returns the first frame of the video. Used for precalculations.
    """
    video.set(1, index); # Where index is the frame you want
    _, frame = video.read() # Read the frame
    image = frame[:, :, 0]
    
    if CONFIG.video_settings.edges:
        image = clip_image(image)
        height, width = image.shape[:2]
        
    if CONFIG.video_settings.height<height and CONFIG.video_settings.height != 1 or CONFIG.video_settings.width>width and CONFIG.video_settings.width !=1:
        image = cropping_image(image, CONFIG.video_settings.height, CONFIG.video_settings.width, CONFIG.video_settings.corner)
    else:
        image = frame[:,:,0]
    return image


def clip_image (arr):
    """
    Some videos have black "edges". This disrupts the phase retrieval, hence we need to clip the images.
    """

    slice_x, slice_y = ndimage.find_objects(arr>0)[0]
    img = arr[slice_x, slice_y]  
    
    #Sometimes due to rounding we get one pixel padded.
    diff_dim = img.shape[0] - img.shape[1]
    if diff_dim == 1:img = img[1:,:] #Change row
    if diff_dim == -1: img = img[:,1:] #Change col
        
    return img

def main():
    c_str = f'Results/{CONFIG.main_settings.project_name}/plots'


    video = cv2.VideoCapture(CONFIG.main_settings.filename_holo) # videobject
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #width 
    downsample_size = CONFIG.plot_settings.downsamplesize

    #Plot the first frame.
    field0 = np.load(f'Results/{CONFIG.main_settings.project_name}/field/field.npy')[0]     
    save_frame(field0.real, c_str, name = 'optical_field_real')
    save_frame(field0.imag, c_str, name = 'optical_field_imag')
    save_frame(np.abs(field0), c_str, name = 'intensity_image')
    save_frame(np.angle(field0), c_str, name = 'phase_image')


    im = first_frame(video, height, width, index=CONFIG.reconstruction_settings.first_frame_precalc)
    im = phase_utils.get_shifted_fft(im, filter_radius = [], correct_fourier_peak=CONFIG.reconstruction_settings().correct_fourier_peak)

    #####Plot fft image
    plt.figure(figsize = (12, 12))
    plt.imshow(np.log(np.abs(im)),cmap = 'gray')    
    plt.axhline(y = CONFIG.video_settings.height/2, linewidth=2, color = 'red') if CONFIG.video_settings.height!= 1 else plt.axhline(y = height/2, linewidth=2, color = 'red')
    plt.axvline(x = CONFIG.video_settings.width/2, linewidth=2, color = 'red') if CONFIG.video_settings.width!= 1 else plt.axvline(x = width/2, linewidth=2, color = 'red')
    plt.savefig(f'{c_str}/1fft_centered.png', bbox_inches = 'tight', dpi = CONFIG.plot_settings.DPI)
 
    ##### Plot all 
    if CONFIG.plot_settings.plot_all:
       field = np.load(f'Results/{CONFIG.main_settings.project_name}/field/field.npy')

       if CONFIG.reconstruction_settings.normalize_field:
            field = [correctfield(fi) for fi in field]

        #Plot all frames
       for i in range(len(field)):
           a = np.concatenate((field[i].real, field[i].imag), axis = 1)
           a = downsample2d(a, downsample_size) # Dwonsample somwehat
           
           save_frame(a, c_str, name = f"frames/real_imag{i}")
       
       #Refocus field and plot.
       if CONFIG.plot_settings.plot_z:    
           l , u = CONFIG.z_propagation_settings.z_search_low, CONFIG.z_propagation_settings.z_search_high
           z = np.linspace(l, u, 21)
           
           if np.abs(CONFIG.z_propagation_settings.z_prop) > 0:
               field0 = Utils_z.refocus_field_z(field0, -CONFIG.z_propagation_settings.z_prop)
           
           for i, zi in enumerate(z):
               fp = Utils_z.refocus_field_z(field0, zi)
               r = downsample2d(fp.real, downsample_size) # Dwonsample somwehat
               save_frame(r, c_str + '/prop', name = f"{i}_z_{np.round(zi, 3)}_real")  
    
       #Plot subtracted data 
       if CONFIG.plot_settings.plot_sub:
           if CONFIG.index_settings.index_method == 'all':
               for i, f in enumerate(field):
                   
                   try:
                       sub = f - field[i + CONFIG.video_settings.vid_shift + CONFIG.index_settings.frame_disp_vid]
                       a = np.concatenate((sub.real, sub.imag), axis = 1)
                       a = downsample2d(a, downsample_size) # Dwonsample somwehat
                       save_frame(a, c_str + '/sub', name = f"real_imag{i}")
                   except:
                       pass

           if CONFIG.index_settings.index_method == 'old' or CONFIG.index_settings.index_method == 'pre2' or CONFIG.index_settings.index_method == 'pre3' or CONFIG.index_settings.index_method == 'pre4' or CONFIG.index_settings.index_method == 'pre5' or CONFIG.index_settings.index_method == 'prepost':
               sub_data = subtract_data(field, CONFIG.index_settings.frame_disp_vid, sign = 1)
               
               for i, sub in enumerate(sub_data):
                   a = np.concatenate((sub.real, sub.imag), axis = 1)
                   a = downsample2d(a, downsample_size) # Dwonsample somwehat
                   save_frame(a, c_str + '/sub', name = f"real_imag{i}")


    #Do movies of some of the plots
    if CONFIG.plot_settings.movie:
        
        #Plot all frames
        image_utils.save_video(folder = c_str + '/frames/', savefolder = c_str + '/frames_movie.avi', fps = CONFIG.plot_settings.movie_fps)

        #Plot subtracted data
        if CONFIG.plot_settings.plot_sub:
            image_utils.save_video(folder = c_str + '/sub/', savefolder = c_str + '/sub_movie.avi', fps = CONFIG.plot_settings.movie_fps)
        
        #Plot refocused data
        if CONFIG.plot_settings.plot_z:
            image_utils.save_video(folder = c_str + '/prop/', savefolder = c_str + '/prop_movie.avi', fps = CONFIG.plot_settings.movie_fps)
 
        
#Main function
if __name__ == "__main__":
    main()