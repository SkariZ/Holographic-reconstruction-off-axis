# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:13:49 2021

@author: Fredrik Skärberg
"""

import CONFIG
CR = CONFIG.reconstruction_settings() # Initialize for array construction
CV = CONFIG.index_settings() # Initialize for array construction


from Utils import phase_utils
from Utils import video_shift_info
from Utils import get_good_frames_idx
from Utils import estimate_vid_shift
from Utils import get_idx
from Utils import Utils_z
from Utils import image_utils as u

import cv2, time, gc, numpy as np, os
from scipy import ndimage

import multiprocessing as mp    

def imgtofield(img, 
               G, 
               polynomial, 
               kx_add_ky,
               if_lowpass_b = False, 
               cropping=50,
               mask_f = [], # sinc, jinc etc.
               z_prop = 0,
               masks = [],
               add_phase_corrections = 0,
               first_phase_background = []
               ):
    
    """ 
    Function for constructing optical field.
   
    """
    #Scale image by its mean
    img = np.array(img, dtype = np.float32) 
    img = img - np.mean(img) #img = img - np.mean(img)
    
    #Compute the 2-dimensional discrete Fourier Transform with offset image.
    fftImage = np.fft.fft2(img * np.exp(1j*(kx_add_ky)))

    #shifted fourier image centered on peak values in x and y. 
    fftImage = np.fft.fftshift(fftImage)
    
    #Sets values outside the defined circle to zero. Ie. take out the information for this peak.
    fftImage2 = fftImage * masks[0] 

    #If we have a weighted function. E.g sinc or jinc.
    if len(mask_f)>0: fftImage2 = fftImage2 * mask_f
    
    #Shift the zero-frequency component to the center of the spectrum.
    E_field = np.fft.fftshift(fftImage2)

    #Inverse 2-dimensional discrete Fourier Transform
    E_field = np.fft.ifft2(E_field) 

    #Removes edges in x and y. Some edge effects
    E_field_cropped = E_field[cropping:-cropping, cropping:-cropping]
    
    #If we use the same first phase background correction on all the data.
    if len(first_phase_background)>0:
        E_field_cropped = E_field_cropped * np.exp( -1j * first_phase_background)

    #Lowpass filtered phase
    if if_lowpass_b:
        phase_img = phase_utils.phase_frequencefilter(fftImage2, mask = masks[1] , is_field = False, crop = cropping) 
    else:
        phase_img  = np.angle(E_field_cropped) #Returns the angle of the complex argument (phase)
    
    # Get the phase background from phase image.
    phase_background = phase_utils.correct_phase_4order(phase_img, G, polynomial)
    E_field_corr = E_field_cropped * np.exp( -1j * phase_background)

    #Do additional background fit. Always lowpass
    if add_phase_corrections>0 and if_lowpass_b:
        for _ in range(add_phase_corrections):
            phase_img = phase_utils.phase_frequencefilter(E_field_corr, mask = masks[2], is_field = True) 
            phase_background = phase_utils.correct_phase_4order(phase_img, G, polynomial)
            E_field_corr =  E_field_corr * np.exp( -1j * phase_background)

    #Lowpass filtered phase
    if if_lowpass_b:
        phase_img2 = phase_utils.phase_frequencefilter(E_field_corr, mask = masks[3], is_field = True)  
    else:
        phase_img2 = np.angle(E_field_corr)
    
    #Correct E_field again
    E_field_corr2 = E_field_corr * np.exp(- 1j * np.median(phase_img2 + np.pi - 1))

    #Focus the field
    if np.abs(z_prop) > 0:  
        E_field_corr2 = Utils_z.refocus_field_z(E_field_corr2, z_prop, padding = 64)
        
    return E_field_corr2

def video_to_field_n(index):
    """
    Function for calling optical field reconstruction. Written such that only input is index of video.
    """
    video.set(1, index); # Where index is the frame you want
    _, frame = video.read() # Read the frame
    
    try: #Some frames might be disrupted...
        tmp_img = frame[:, :, 0]
        
        if CONFIG.video_settings.edges:
            tmp_img = u.clip_image(tmp_img)
            height, width = tmp_img.shape[:2]
            
        if CONFIG.video_settings.height<height and CONFIG.video_settings.height != 1 or CONFIG.video_settings.width>width and CONFIG.video_settings.width !=1:
            tmp_img = u.cropping_image(tmp_img, CONFIG.video_settings.height, CONFIG.video_settings.width, CONFIG.video_settings.corner)
        
        # Retrieve optical field and phase background.
        E_field = imgtofield(tmp_img, 
                             G, 
                             polynomial, 
                             kx_add_ky,
                             if_lowpass_b = CONFIG.reconstruction_settings.lowpass_fit, 
                             cropping=CONFIG.reconstruction_settings.cropping,  
                             mask_f = [],
                             z_prop = CONFIG.z_propagation_settings.z_prop,
                             masks = masks,
                             add_phase_corrections= CONFIG.reconstruction_settings.add_phase_corrections,
                             first_phase_background = phase_background
                             )
    except:
        E_field = np.zeros((first_img.shape))
        E_field = np.array(E_field, dtype = np.complex64)
        
    return E_field


#####GLOBAL VARIABLES##### (necessary for multiprocessing)
video = cv2.VideoCapture(CONFIG.main_settings.filename_holo) # videobject
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of frames
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #width
fps = int(video.get(cv2.CAP_PROP_FPS)) #fps
#first_img = first_frame(video, height, width, index=CONFIG.reconstruction_settings.first_frame_precalc)
first_img = u.first_frame(video, height, width,CONFIG.video_settings.height, CONFIG.video_settings.width, index=CONFIG.reconstruction_settings.first_frame_precalc)

#Precalculations (Speed ups computations a lot).
X, Y, X_c, Y_c, position_matrix, G, polynomial, KX, KY, KX2_add_KY2, kx_add_ky, dist_peak, masks, phase_background, rad  = phase_utils.pre_calculations(
    first_img, 
    filter_radius = [], 
    cropping = CONFIG.reconstruction_settings.cropping, 
    mask_radie = CR.radius_lowpass, 
    case = 'circular', 
    first_phase_background = CONFIG.reconstruction_settings.first_phase_background)

jinc_mask = phase_utils.jinc(position_matrix / dist_peak / 3)
sinc_mask = np.sinc(position_matrix / dist_peak / 3)

#Search for focus on first frame.
if CONFIG.z_propagation_settings.find_focus_first_frame and np.abs(CONFIG.z_propagation_settings.z_prop) == 0:
    if np.abs(CONFIG.z_propagation_settings.find_focus_first_frame_idx_start - CONFIG.z_propagation_settings.find_focus_first_frame_idx_stop) == 0:
        field = video_to_field_n(CONFIG.reconstruction_settings.first_frame_precalc)
    else:
        field = video_to_field_n(
            CONFIG.z_propagation_settings.find_focus_first_frame_idx_start) - video_to_field_n(CONFIG.z_propagation_settings.find_focus_first_frame_idx_stop
            )
    CONFIG.z_propagation_settings.z_prop, criterion = Utils_z.find_focus_field(field, steps=51, interval = [CONFIG.z_propagation_settings.z_search_low, CONFIG.z_propagation_settings.z_search_high], m = 'abs', bbox = [300, 600, 300, 600], use_max_real=True)
    
def main():
    print("Original video size: (frames, height, width, fps):", (n_frames, height, width, fps))
    print('Focus found: ', np.round(CONFIG.z_propagation_settings.z_prop, 3))


    # Predefined video / camera settings
    video_shift_data = video_shift_info.get_video_shift_info()

    #Check if frames are disrupted and/or very different
    if CONFIG.video_settings.check_good:
        print("Checking for disrupted, black frames, and/or very different frames...")
        good_indexes = get_good_frames_idx.good_frames(video)
    else:
        good_indexes = np.array(np.arange(0, n_frames))
    
    #Check if videosettings are available
    if video_shift_info.check_if_video_settings_available(video_shift_data, CONFIG.main_settings.filename_holo):
        print("Predefined camera settings found")
        CONFIG.video_settings.vid_shift, CONFIG.video_settings.frame_disp_vid = video_shift_info.return_available_video_settings(video_shift_data, CONFIG.main_settings.filename_holo)
    elif CONFIG.video_settings.vid_shift !=0:
        print("Manual video shift inputed...")
        pass
    else:     
        print("Predefined camera settings NOT found. Estimating video shift...")
        CONFIG.video_settings.vid_shift = estimate_vid_shift.estimate_peak_video(video, CONFIG.video_settings.height, CONFIG.video_settings.width, good_indexes[:200], median = False)    
        print("Video shift estimated to be: ", CONFIG.video_settings.vid_shift)
    
    #Indexes to extract.
    index_method = CONFIG.index_settings.index_method
    input_mp = get_idx.get(good_indexes, 
                           start_frame = CONFIG.index_settings.start_frame, 
                           vid_shift = CONFIG.video_settings.vid_shift, 
                           frame_disp_vid = CONFIG.index_settings.frame_disp_vid, 
                           max_frames = CONFIG.index_settings.max_frames,
                           method = index_method,
                           every = CONFIG.video_settings.vid_shift,
                           index = CV.index)

    print("Number of frames to be extracted...:", len(input_mp))
    
    #Run multiprocessing
    if CONFIG.multiprocessing.M: 
        print("Running multiprocessing")
        pool = mp.Pool(processes=mp.cpu_count() - 2)#, maxtasksperchild=1000)
        start_time = time.time()
        time.sleep(5)
        field = pool.map(video_to_field_n, list(input_mp))
        print("Field extracted...")
        print("--- Time per frame %s seconds ---" % ((time.time() - start_time) / len(input_mp)))
        time.sleep(5)
        
    #Run without multiprocessing    
    else:
        print("Running without multiprocessing")    
        field = []
        start_time = time.time()
        for i, frame in enumerate(input_mp):
            curr_field = video_to_field_n(frame)  
            field.append(curr_field)
            print("Current frame: ", i, "/", len(input_mp))
        print("--- Time per frame %s seconds ---" % ((time.time() - start_time) / len(input_mp)))
        
    #Field is a complex numpy array
    field = np.array(field, dtype = np.complex64)

    f'Results/{CONFIG.main_settings.project_name}/field/field.npy' 
    #Save field and indexes to file
    np.save(f'Results/{CONFIG.main_settings.project_name}/field/field.npy', field)
    np.save(f'Results/{CONFIG.main_settings.project_name}/field/idx.npy', input_mp)

    gc.collect()
    del field

#Main function
if __name__ == "__main__":
    main()



