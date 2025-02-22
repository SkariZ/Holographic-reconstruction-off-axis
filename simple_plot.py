# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:23:54 2021

@author: Fredrik Skärberg
"""
import imageio.v2 as imageio
import CONFIG

import matplotlib.pyplot as plt
import cv2, os
import cupy as cp

from Utils import phase_utils
from Utils import Utils_z
from Utils import image_utils
from Utils import fft_loader

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

def main(
        filename_holo,
        project_name,
        root_folder):
    
    c_str = f'{root_folder}/{project_name}/plots'

    #Settings
    video = cv2.VideoCapture(filename_holo) # videobject
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #width 
    downsample_size = CONFIG.plot_settings.downsamplesize

    #Plot the first frame.
    if CONFIG.save_settings.fft_save:
        field0 = fft_loader.vec_to_field_multi(
                vecs = cp.load(f'{root_folder}/{project_name}/field/field.npy')[0:1], 
                shape = (CONFIG.video_settings.height - 2*CONFIG.reconstruction_settings.cropping, CONFIG.video_settings.width - 2*CONFIG.reconstruction_settings.cropping),
                pupil_radius=CONFIG.save_settings.pupil_radius
            )[0]
    else:
        field0 = cp.load(f'{root_folder}/{project_name}/field/field.npy')[0]

    image_utils.save_frame(field0.real, c_str, name = 'optical_field_real')
    image_utils.save_frame(field0.imag, c_str, name = 'optical_field_imag')
    image_utils.save_frame(cp.abs(field0), c_str, name = 'intensity_image')
    image_utils.save_frame(
        cp.angle(phase_utils.phaseunwrap_skimage(field0, norm_phase_after=True)), 
        c_str, name = 'phase_image'
        )

    im = image_utils.first_frame(video, height, width, CONFIG.video_settings.height, CONFIG.video_settings.width, corner=CONFIG.video_settings.corner, index=CONFIG.reconstruction_settings.first_frame_precalc)
    im = cp.asarray(im)
    im = phase_utils.get_shifted_fft(im, filter_radius = [], correct_fourier_peak=CONFIG.reconstruction_settings().correct_fourier_peak)

    #####Plot fft image
    plt.figure(figsize = (8, 8))
    plt.imshow(cp.log(cp.abs(im)).get(),cmap = 'gray')    
    plt.axhline(y = CONFIG.video_settings.height/2, linewidth=2, color = 'red') if CONFIG.video_settings.height!= 1 else plt.axhline(y = height/2, linewidth=2, color = 'red')
    plt.axvline(x = CONFIG.video_settings.width/2, linewidth=2, color = 'red') if CONFIG.video_settings.width!= 1 else plt.axvline(x = width/2, linewidth=2, color = 'red')
    plt.savefig(f'{c_str}/1fft_centered.png', bbox_inches = 'tight', dpi = CONFIG.plot_settings.DPI, facecolor = 'white')
    plt.close()

    ##### Plot all 
    if CONFIG.plot_settings.plot_all:
       field = cp.load(f'{root_folder}/{project_name}/field/field.npy')

       #Load field if fft_save is true
       if CONFIG.save_settings.fft_save:
            field = fft_loader.vec_to_field_multi(
                    vecs = field, 
                    shape = (CONFIG.video_settings.height - 2*CONFIG.reconstruction_settings.cropping, CONFIG.video_settings.width - 2*CONFIG.reconstruction_settings.cropping),
                    pupil_radius=CONFIG.save_settings.pupil_radius
                )

        #Plot all frames
       for i in range(len(field)):
           #field[i] = Utils_z.refocus_field_z(field[i], 3.2, padding = 128)
           a = cp.concatenate((field[i].real, field[i].imag), axis = 1)
           a = image_utils.downsample2d(a, downsample_size) # Downsample somwehat
           #a = cp.abs(field[i])
           #Save frame via annotation, else plt.imsave
           if CONFIG.plot_settings.annotate:
                image_utils.save_frame(a, c_str + '/frames', name = f"real_imag{i}", annotate = True, annotatename = f"Frame {i}", dpi = CONFIG.plot_settings.DPI) 
           else:
                image_utils.save_frame(a, c_str, name = f"frames/real_imag{i}")
       
       #Refocus field and plot.
       if CONFIG.plot_settings.plot_z:    
           l , u = CONFIG.z_propagation_settings.z_search_low, CONFIG.z_propagation_settings.z_search_high
           z = cp.linspace(l, u, CONFIG.z_propagation_settings.z_steps)
           
           if cp.abs(CONFIG.z_propagation_settings.z_prop) > 0:
               field0 = Utils_z.refocus_field_z(field0, -CONFIG.z_propagation_settings.z_prop, padding = 256)
           
           for i, zi in enumerate(z):
               fp = Utils_z.refocus_field_z(field0, zi, padding = 128)
               r = cp.concatenate((fp.real, fp.imag), axis = 1)
               r = image_utils.downsample2d(r, downsample_size) # Downsample somwehat

                #Save frame via annotation, else plt.imsave
               if CONFIG.plot_settings.annotate:
                    image_utils.save_frame(r, c_str + '/prop', name = f"{i}_z_{cp.round(zi, 3)}", annotate = True, annotatename = f"z = {cp.round(zi, 3)}", dpi = CONFIG.plot_settings.DPI)
               else:
                    image_utils.save_frame(r, c_str + '/prop', name = f"{i}_z_{cp.round(zi, 3)}")
    
       #Plot subtracted data 
       if CONFIG.plot_settings.plot_sub:
           if CONFIG.index_settings.index_method == 'all':
               for i, f in enumerate(field):
                   
                   try:
                       sub = f - field[i + CONFIG.video_settings.vid_shift + CONFIG.index_settings.frame_disp_vid]
                       a = cp.concatenate((sub.real, sub.imag), axis = 1)
                       a = image_utils.downsample2d(a, downsample_size) # Downsample somwehat
                       image_utils.save_frame(a, c_str + '/sub', name = f"real_imag{i}")
                   except:
                       pass

           if CONFIG.index_settings.index_method == 'old' or CONFIG.index_settings.index_method == 'pre2' or CONFIG.index_settings.index_method == 'pre3' or CONFIG.index_settings.index_method == 'pre4' or CONFIG.index_settings.index_method == 'pre5' or CONFIG.index_settings.index_method == 'prepost':
               sub_data = subtract_data(field, CONFIG.index_settings.frame_disp_vid, sign = 1)
               
               for i, sub in enumerate(sub_data):
                   a = cp.concatenate((sub.real, sub.imag), axis = 1)
                   a = image_utils.downsample2d(a, downsample_size) # Dwonsample somwehat
                   image_utils.save_frame(a, c_str + '/sub', name = f"real_imag{i}")

    #Do movies of some of the plots
    if CONFIG.plot_settings.movie and CONFIG.plot_settings.plot_all:
        
        #Plot all frames
        image_utils.gif(folder = c_str + '/frames/', savefolder = c_str + '/frames_movie.gif')

        #Plot subtracted data
        if CONFIG.plot_settings.plot_sub:
            image_utils.gif(folder = c_str + '/sub/', savefolder = c_str + '/sub_movie.gif')
            #image_utils.save_video(folder = c_str + '/sub/', savefolder = c_str + '/sub_movie.avi', fps = CONFIG.plot_settings.movie_fps)
        
        #Plot refocused data
        if CONFIG.plot_settings.plot_z:
            image_utils.gif(folder = c_str + '/prop/', savefolder = c_str + '/prop_movie.gif')
            #image_utils.save_video(folder = c_str + '/prop/', savefolder = c_str + '/prop_movie.avi', fps = CONFIG.plot_settings.movie_fps)
 
    #Delete all images after the movie is constructed
    if CONFIG.plot_settings.delete_images:
        #Delete all images in frames folder
        for filename in os.listdir(c_str + '/frames/'):
            os.remove(c_str + '/frames/' + filename)
        
        #Delete all images in sub folder
        if CONFIG.plot_settings.plot_sub:
            for filename in os.listdir(c_str + '/sub/'):
                os.remove(c_str + '/sub/' + filename)

        #Delete all images in prop folder
        if CONFIG.plot_settings.plot_z:
            for filename in os.listdir(c_str + '/prop/'):
                os.remove(c_str + '/prop/' + filename)

#Main function
if __name__ == "__main__":
    main()