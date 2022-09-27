# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:17:50 2021

@author: Fredrik Skärberg
"""

import cv2, numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from scipy import signal

def cropping_image(image, h, w):
    wi, hi = image.shape[:2]
    if wi<w or hi<h:
        raise Exception("Cropping size larger than actual image size.")
    
    if wi == hi:#If we have a square image we can resize without loss of quality
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
    else: #Crop out the "corner"
        image = image[:h, :w]

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
    if diff_dim == -1: img = img[:,1:]#Change col
        
    return img

def reject_outliers_std(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def estimate_peak_video(video, H, W, indexes = np.arange(0, 200), lag_comp = 1, distance_search_period = 10, median = False, reject_outliers=True):
    """
        Estimates the period of which the camera shiftes. Using cross corelation.
    """
    frames = []
    for index in indexes:
        video.set(1, index); # Where index is the frame you want
        _, frame = video.read() # Read the frame
        tmp_img = frame[:,:,0]
        height, width = tmp_img.shape[:2]
        
        #Check for black edges
        tmp_img = clip_image(tmp_img)
        height, width = tmp_img.shape[:2]
        
        if W<height and H != 1 or H>width and W !=1:
            tmp_img = cropping_image(tmp_img, H, W)
        
        #check for black frames
        if np.sum(tmp_img==0)< (tmp_img.shape[0] * tmp_img.shape[1]):
            frames.append(tmp_img)
    frames = np.array(frames, dtype = np.float32)
    
    frames = np.array([frame/np.mean(frame) for frame in frames], dtype = np.float32)

    #https://scikit-image.org/docs/0.11.x/auto_examples/plot_register_translation.html
    #Cross correlation
    errors = []
    for i in range(lag_comp, len(frames)):
        _, error, _ = phase_cross_correlation(
            frames[i-lag_comp], 
            frames[i], 
            space='fourier',
            normalization='phase') #FFT
        errors.append(error)
    errors = np.array(errors)     
    
    #Identify peaks
    # distance = Required minimal horizontal distance
    peaks = signal.find_peaks(errors, height = np.median(errors), distance = distance_search_period)
    peak_diff = np.diff(peaks[0], n = 1)

    #Remove "weird" observations
    if reject_outliers:
        peak_diff = reject_outliers_std(peak_diff, m = 1.75)

    if median:
        period = int(np.median(peak_diff)) #The most common period.
    else:
        period = np.mean(peak_diff) #Mean over the periods.

    start_frame = find_start_frame(errors, distance_search_range = np.arange(int(period/2), int(3/2 * period)))
    
    return period, start_frame+lag_comp

def estimate_peak_array(frames, max_ind = 1, lag_comp = 1, distance_search_period = 10, median = False, reject_outliers=True):
    """
        Estimates the period of which the camera shiftes. Using cross corelation.
    """

    if np.iscomplexobj(frames):
        frames = np.array([frame/(np.mean(frame.real) + 1j*np.mean(frame.imag)) for frame in frames], dtype = np.complex64)
    else:
        frames = np.array([frame/np.mean(frame) for frame in frames], dtype = np.float32)

    if max_ind !=1 and max_ind > 0:
        frames = frames[:max_ind]
    
    #https://scikit-image.org/docs/0.11.x/auto_examples/plot_register_translation.html
    #Cross correlation
    errors = []
    for i in range(lag_comp, len(frames)):
        _, error, _ = phase_cross_correlation(
            frames[i-lag_comp], 
            frames[i], 
            space='fourier',
            normalization='phase') #FFT
        errors.append(error)

    errors = np.array(errors)    
    
    #Identify peaks
    # distance = Required minimal horizontal distance
    peaks = signal.find_peaks(errors, height = np.median(errors), distance = distance_search_period)
    peak_diff = np.diff(peaks[0], n = 1)
    
    #Remove "weird" observations
    if reject_outliers:
        peak_diff = reject_outliers_std(peak_diff, m = 1.75)

    if median:
        period = int(np.median(peak_diff)) #The most common period.
    else:
        period = np.mean(peak_diff) #Mean over the periods.
    
    start_frame = find_start_frame(errors, distance_search_range = np.arange(int(period/2), int(3/2 * period)))

    return period, start_frame+lag_comp


def find_start_frame(errors, distance_search_range = [5, 10, 15, 20]):
    peaks_potential = []
    for distance in distance_search_range:
        peaks_potential.append(
            signal.find_peaks(
            errors, 
            height = np.median(errors),
            distance = distance
            )[0][0]
        )



"""
f0 = f[7] / (np.median(f[7].real) + 1j * np.median(f[7].imag))
f1 = f[13] / (np.median(f[13].real) + 1j * np.median(f[13].imag))

from matplotlib_scalebar.scalebar import ScaleBar

real_fields = [f0.real, f1.real, (f0-f1).real]
imag_fields = [f0.imag, f1.imag, (f0-f1).imag]

scalebar = ScaleBar(0.0114, units = 'µm', color = 'red', location = 'lower right')

fig, ax = plt.subplots(nrows=2, ncols=3)

for i, r in enumerate(real_fields):
    scalebar = ScaleBar(0.0114, units = 'µm', color = 'red', location = 'lower right')
    ax[0, i].set_xticks([])
    ax[0, i].set_yticks([])
    
    ax[0, i].imshow(r, cmap = 'gray')
    
    if i == 0:
        ax[0, i].set_title(f"A")
        ax[0, i].set_ylabel('Real part')
    elif i == 1:
        ax[0, i].set_title(f"B")
    else:
        ax[0, i].set_title(f"A - B")
    ax[0, i].add_artist(scalebar)
    
for i, r in enumerate(imag_fields):
    scalebar = ScaleBar(0.0114, units = 'µm', color = 'red', location = 'lower right')
    ax[1, i].set_xticks([])
    ax[1, i].set_yticks([])
    
    ax[1, i].imshow(r, cmap = 'gray')
    ax[1, i].add_artist(scalebar)
    if i == 0:
        ax[1, i].set_ylabel('Imaginary part')


fig.savefig('b.png', dpi = 200, bbox_inches='tight')

#plt.figure()
#plt.imshow(first_img[50:-50, 50:-50], cmap = 'gray')
#plt.axis('off')
"""


"""  
tmp_field = []
for i in range(len(f)-2):
    tmp = f[i] - f[i+2]
    
    tmp_field.append(tmp)
    
def save_frame(frame, folder, name, cmap = 'gray'):
    plt.imsave(f"{folder}/{name}.png", frame, cmap = 'gray')

c_str = 'Results/' + DATA.project_name + '/sub_plots'
for i in range(len(tmp_field)):
       a = np.concatenate((tmp_field[i].real, tmp_field[i].imag), axis = 1)
       save_frame(a, c_str, name = f"real_imag{i}")
     


import scipy.signal
from scipy.signal import convolve2d
kernel=np.ones((32,32))/(32*32)

def preprocess_field_conv(image0, kernel):
    
    an0=np.angle(image0*np.exp(-1j))+1
    ang0=convolve2d(an0, kernel, mode="same")
    
    image0/=np.mean(np.abs(image0))
    image0*=np.exp(-1j*(ang0))
    
    return image0         
"""
"""
def save_frame(frame, folder, name, cmap = 'gray'):
    plt.imsave(f"{folder}/{name}.png", frame, cmap = 'gray')
    
c_str = 'Results/' + DATA.project_name + '/prop_plots'
for i, z in enumerate(np.linspace(-10, 10, 201)):
    
    tmp_field = phase_utils.refocus_field_z(field, z)
    a = np.concatenate((tmp_field.real, tmp_field.imag), axis = 1)
    save_frame(a, c_str, name = f"real_imag{i}_{np.round(z, 3)}")
    print(i)
    
    
    


def plot_text_on_png_cv2(image_path, text_position = (10,100), text = "", color = (255,255,255), font_size = 3):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  
    
    cv2.putText(
         image, #numpy array on which text is written
         text, #text
         text_position, #position at which writing has to start
         cv2.FONT_HERSHEY_SIMPLEX, #font family
         font_size, #font size
         color, #font color
         3) #font stroke

    cv2.imwrite(image_path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    
    
    
import cv2

for i, z in enumerate(np.linspace(-10, 10, 201)):

    image = cv2.imread(f"{c_str}/real_imag{i}_{np.round(z, 3)}.png", cv2.IMREAD_UNCHANGED)
    
    position = (10,100)
    cv2.putText(
         image, #numpy array on which text is written
         f"Z_prop = {np.round(z, 3)}", #text
         position, #position at which writing has to start
         cv2.FONT_HERSHEY_SIMPLEX, #font family
         3, #font size
         (255,255,255), #font color
         3) #font stroke
    
    cv2.imwrite(f"{c_str}/real_imag{i}_{np.round(z, 3)}.png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
 """