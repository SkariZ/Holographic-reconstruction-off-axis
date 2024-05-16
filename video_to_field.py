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

import cv2, time, gc
import cupy as cp
import matplotlib.pyplot as plt

def imgtofield(img, 
               G, 
               polynomial, 
               kx_add_ky,
               X,
               Y,
               if_lowpass_b = False, 
               cropping=50,
               mask_f = [], # sinc, jinc etc.
               z_prop = 0,
               masks = [],
               add_phase_corrections = 0,
               first_phase_background = [],
               correct_phase_background_tol = 0
               ):
    
    """ 
    Function for constructing optical field.
   
    """
    #Scale image by its mean
    img = cp.array(img, dtype = cp.float32) 
    img = img - cp.mean(img) #img = img - cp.mean(img)

    #Compute the 2-dimensional discrete Fourier Transform with offset image.
    fftImage = cp.fft.fft2(img * cp.exp(1j*(kx_add_ky)))

    #shifted fourier image centered on peak values in x and y. 
    fftImage = cp.fft.fftshift(fftImage)

    #Sets values outside the defined circle to zero. Ie. take out the information for this peak.
    fftImage2 = fftImage * masks[0] 
    
    #If we have a weighted function. E.g sinc or jinc.
    if len(mask_f)>0: fftImage2 = fftImage2 * mask_f
    
    #Shift the zero-frequency component to the center of the spectrum.
    E_field = cp.fft.fftshift(fftImage2)

    #Inverse 2-dimensional discrete Fourier Transform
    E_field = cp.fft.ifft2(E_field) 

    #Removes edges in x and y. Some edge effects
    if cropping>0:
        E_field_cropped = E_field[cropping:-cropping, cropping:-cropping]
    else:
        E_field_cropped = E_field
    
    #If we use the same first phase background correction on all the data.
    if len(first_phase_background)>0:
        E_field_cropped = E_field_cropped * cp.exp( -1j * first_phase_background)
    
    #Lowpass filtered phase
    if if_lowpass_b:
        phase_img = phase_utils.phase_frequencefilter(fftImage2, mask = masks[1] , is_field = False, crop = cropping) 
    else:
        phase_img  = cp.angle(E_field_cropped) #Returns the angle of the complex argument (phase)
    
    # Get the phase background from phase image.
    phase_background = phase_utils.correct_phase_4order(phase_img, G, polynomial)
    E_field_corr = E_field_cropped * cp.exp( -1j * phase_background)

    #Do additional background fits.
    if add_phase_corrections>0:
        for _ in range(add_phase_corrections):
            if if_lowpass_b:
                phase_img = phase_utils.phase_frequencefilter(E_field_corr, mask = masks[2], is_field = True)
            else:
                phase_img = cp.angle(E_field_corr)

            phase_background = phase_utils.correct_phase_4order(phase_img, G, polynomial)
            E_field_corr =  E_field_corr * cp.exp( -1j * phase_background)
    
    #Do a while loop to correct the phase background, until it is close to zero.
    if correct_phase_background_tol > 0:
        counter = 0
        while cp.abs(cp.mean(phase_background)) > correct_phase_background_tol:
            counter += 1
            if if_lowpass_b:
                phase_img = phase_utils.phase_frequencefilter(E_field_corr, mask = masks[2], is_field = True)
            else:
                phase_img = cp.angle(E_field_corr)

            phase_background = phase_utils.correct_phase_4order(phase_img, G, polynomial)
            E_field_corr =  E_field_corr * cp.exp( -1j * phase_background)

            if counter > 20:
                print("Correcting phase background while loop is not converging. Breaking loop...")
                break

    #Lowpass filtered phase
    if if_lowpass_b:
        phase_img2 = phase_utils.phase_frequencefilter(E_field_corr, mask = masks[3], is_field = True)  
    else:
        phase_img2 = cp.angle(E_field_corr)

    #Correct E_field again
    E_field_corr2 = E_field_corr * cp.exp(- 1j * cp.mean(phase_img2))
    
    #Focus the field if z_prop is not zero.
    if cp.abs(z_prop) > 0:  
        E_field_corr2 = Utils_z.refocus_field_z(E_field_corr2, z_prop, padding = 512)
        
    return E_field_corr2

def video_to_field(
        video,
        index,
        G,
        polynomial,
        kx_add_ky,
        X,
        Y,
        mask_f = [],
        masks = [],
        first_phase_background = []
        ):
    """
    Function for calling optical field reconstruction. Written such that only input is index of video.
    """
    video.set(1, index); # Where index is the frame you want
    _, frame = video.read() # Read the frame
    
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
                            X,
                            Y,
                            if_lowpass_b = CONFIG.reconstruction_settings.lowpass_fit,
                            #unwrap = CONFIG.reconstruction_settings.unwrap, 
                            cropping = CONFIG.reconstruction_settings.cropping,  
                            mask_f = mask_f,
                            z_prop = CONFIG.z_propagation_settings.z_prop,
                            masks = masks,
                            add_phase_corrections= CONFIG.reconstruction_settings.add_phase_corrections,
                            first_phase_background = first_phase_background,
                            correct_phase_background_tol = CONFIG.reconstruction_settings.correct_phase_background_tol
                            )
    return E_field

def main(filename_holo,
         project_name,
         root_folder):

    video = cv2.VideoCapture(filename_holo) # videobject
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of frames
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #width
    fps = int(video.get(cv2.CAP_PROP_FPS)) #fps
    first_img = u.first_frame(video, height, width, CONFIG.video_settings.height, CONFIG.video_settings.width, index=CONFIG.reconstruction_settings.first_frame_precalc)
    first_img = cp.array(first_img)
    print("Original video size: (frames, height, width, fps):", (n_frames, height, width, fps))
    
    #Precalculations (Speed ups computations a lot).
    _, _, X, Y, position_matrix, G, polynomial, _, _, _, kx_add_ky, dist_peak, masks, phase_background, _  = phase_utils.pre_calculations(
        first_img, 
        filter_radius = [], 
        cropping = CONFIG.reconstruction_settings.cropping, 
        mask_radie = CR.radius_lowpass, 
        case = 'ellipse', 
        first_phase_background = CONFIG.reconstruction_settings.first_phase_background,
        mask_out = CONFIG.reconstruction_settings.mask_out)

    #Jinc and sinc masks
    if CONFIG.reconstruction_settings.mask_f == 'jinc':
        mask_f = phase_utils.jinc(position_matrix / dist_peak / 3)
    elif CONFIG.reconstruction_settings.mask_f == 'sinc':
        mask_f = cp.sinc(position_matrix / dist_peak / 3)
    else:
        mask_f = []

    #Search for focus on first frame.
    if CONFIG.z_propagation_settings.find_focus_first_frame and cp.abs(CONFIG.z_propagation_settings.z_prop) == 0:
        if cp.abs(CONFIG.z_propagation_settings.find_focus_first_frame_idx_start - CONFIG.z_propagation_settings.find_focus_first_frame_idx_stop) == 0:
            field = video_to_field(
                video = video,
                index = CONFIG.reconstruction_settings.first_frame_precalc,
                G = G,
                polynomial = polynomial,
                kx_add_ky = kx_add_ky,
                X=X,
                Y=Y,
                mask_f = mask_f,
                masks = masks,
                first_phase_background = phase_background)
        else:
            f1 = video_to_field(
                video = video,
                index = CONFIG.z_propagation_settings.find_focus_first_frame_idx_start,
                G = G,
                polynomial = polynomial,
                kx_add_ky = kx_add_ky,
                X=X,
                Y=Y,
                mask_f = mask_f,
                masks = masks,
                first_phase_background = phase_background)  
              
            f2 =  video_to_field(
                video = video,
                index = CONFIG.z_propagation_settings.find_focus_first_frame_idx_stop,
                G = G,
                polynomial = polynomial,
                kx_add_ky = kx_add_ky,
                mask_f = mask_f,
                masks = masks,
                first_phase_background = phase_background)
            
            field = f1-f2

        #Find focus 
        CONFIG.z_propagation_settings.z_prop, criterion = Utils_z.find_focus_field(
            field, 
            steps=CONFIG.z_propagation_settings.z_steps, 
            interval = [CONFIG.z_propagation_settings.z_search_low, CONFIG.z_propagation_settings.z_search_high], 
            m = 'abs',
            ma=11,
            )
        print('Focus is set to: ', cp.round(CONFIG.z_propagation_settings.z_prop, 3))
    
    # Predefined video / camera settings
    video_shift_data = video_shift_info.get_video_shift_info()

    #Check if frames are disrupted and/or very different
    if CONFIG.video_settings.check_good:
        print("Checking for disrupted, black frames, and/or very different frames...")
        good_indexes = get_good_frames_idx.good_frames(video)
    else:
        good_indexes = cp.array(cp.arange(0, n_frames))
    
    #Check if videosettings are available
    if video_shift_info.check_if_video_settings_available(video_shift_data, filename_holo):
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
    input_mp = get_idx.get(
        good_indexes.get(), 
        start_frame = CONFIG.index_settings.start_frame, 
        vid_shift = CONFIG.video_settings.vid_shift, 
        frame_disp_vid = CONFIG.index_settings.frame_disp_vid, 
        max_frames = CONFIG.index_settings.max_frames,
        method = index_method,
        every = CONFIG.video_settings.vid_shift,
        index = CV.index
        )
    print("Number of frames to be extracted...:", len(input_mp))
    
    field = []
    start_time = time.time()
    for i, frame in enumerate(input_mp):
        curr_field = video_to_field(
            video = video,
            index = frame,
            G = G,
            polynomial = polynomial,
            kx_add_ky = kx_add_ky,
            X=X,
            Y=Y,
            mask_f = mask_f,
            masks = masks,
            first_phase_background = phase_background)  
        
        field.append(curr_field)
        if i % 10 == 0:
            print("Current frame: ", i, "/", len(input_mp))
    print("--- Time per frame %s seconds ---" % ((time.time() - start_time) / len(input_mp)))
        
    #Field is a complex numpy array
    field = cp.array(field, dtype = cp.complex64)

    if CONFIG.reconstruction_settings.conjugate_check:
        field = u.conjugate_check(field)

    if CONFIG.reconstruction_settings.sign_correction:
        field = u.correctfield_sign(field)

    if CONFIG.reconstruction_settings.background_segmentation:
        seg = u.background_segmentation(field, n_frames = 5)
        field = u.background_normalize(field, seg)
        cp.save(f'{root_folder}/{project_name}/field/seg.npy', cp.asarray(seg))

    if CONFIG.reconstruction_settings.normalize_field:
        field = u.correctfield(field, n_iter = 5)

    #Compress field with fft
    if CONFIG.save_settings.fft_save:
        from Utils import fft_loader
        field = fft_loader.field_to_vec_multi(fields = field, pupil_radius=CONFIG.save_settings.pupil_radius)

    #Save field and indexes to file
    cp.save(f'{root_folder}/{project_name}/field/field.npy', field)
    cp.save(f'{root_folder}/{project_name}/field/idx.npy', input_mp)

    #Plot the fourier selection filter.
    a = u.downsample2d(masks[0], CONFIG.plot_settings.downsamplesize) # Downsample somwehat
    u.save_frame(a, f'{root_folder}/{project_name}/plots', name = 'fourier_selection_mask')

    if CONFIG.reconstruction_settings.background_segmentation:
        u.save_frame(cp.asarray(seg[0]), f'{root_folder}/{project_name}/plots', name = 'segmentation_mask')

    gc.collect()
    del field

    #Set z_prop to zero
    CONFIG.z_propagation_settings.z_prop = 0

#Main function
if __name__ == "__main__":
    main()



