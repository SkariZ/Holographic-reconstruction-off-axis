"""
Script contains functions for retrieving optical field from interference pattern obtained in off-axis holography.

Created on Wed Mar 03 09:14:39 2022

@author: Fredrik Skärberg // fredrik.skarberg@physics.gu.se

"""

import numpy as np
from skimage import morphology
from skimage.restoration import unwrap_phase
#import cv2
import scipy
from scipy import ndimage

def correct_phase_4order (phase_img, G, polynomial):
    """ 
    Calculates the coefficients (4th order) by taking the derivative of phase image to fit an phase background. 

    Input: 
        phase_img : phase image
        G : Matrix that store 4-order polynominal
        polynomial : polynomial matrix of meshes
    Output: 
        Phase bakground fit

    """
    An0 = phase_img.copy()
    yr, xr = An0.shape

    An0 = An0 - An0[0, 0] #Set phase to "0"

    # Derivative the phase to handle the modulus
    dx = -np.pi + np.mod(np.pi + (np.diff(An0, axis = 0)), 2*np.pi)
    dy = -np.pi + np.transpose(np.mod(np.pi + np.transpose((np.diff(An0))), 2*np.pi))
    
    #dx1, dy1 the derivatives
    dx1 = 1/2 * ((dx[:, 1:] + dx[:, :-1])).flatten(order='F')
    dy1 = 1/2 * ((dy[1:, :] + dy[:-1, :])).flatten(order='F')

    #(derivate w.r.t to X and Y respectively.) Each factor have a constant b_i to be fitted later on.
    G_end = G.shape[0]
    uneven_range = np.arange(1, G_end+1, 2)
    even_range = np.arange(0, G_end, 2)
    
    dt = np.zeros(2 * (xr-1) * (yr-1))
    dt[even_range] = dx1
    dt[uneven_range] = dy1          
    
    # Here the coefficients to the polonomial are calculated. Note that np.linalg.lstsq(b,B)[0] is equivivalent to \ in MATLAB              
    R = np.transpose(np.linalg.cholesky(np.matmul(np.transpose(G),G)))
    
    # Equivivalent to R\(R'\(G'*dt))
    b = np.linalg.lstsq(R, 
                        np.linalg.lstsq(np.transpose(R),
                                        np.matmul(np.transpose(G),dt), rcond=None)[0], rcond=None)[0]
    
    # Phase background is defined by the 4th order polynomial with the fitted parameters.
    phase_background = 0
    for i, factor in enumerate(polynomial):
        phase_background = phase_background + b[i]*factor
    
    return phase_background

def phaseunwrap_skimage(field, norm_phase_after = True):
    """
    Unwrap the phase of the field using skimage.

    Input:
        field : Complex field
    Output:
        Corrected field
    """

    #Check if the field is complex
    if np.iscomplexobj(field):
        phase = np.angle(field)
    else:
        raise ValueError("The field is not complex")

    #Unwrap the phase
    phase_unwrapped = unwrap_phase(phase)

    #Normalize the phase to be between -pi and pi after unwrapping
    if norm_phase_after:
        phase_unwrapped = phase_unwrapped - np.min(phase_unwrapped)
        phase_unwrapped = phase_unwrapped / np.max(phase_unwrapped)
        phase_unwrapped = phase_unwrapped * 2*np.pi - np.pi

    return np.abs(field)*np.exp(1j * phase_unwrapped)


def phase_unwrap_pipeline(field, X_c, Y_c, polynomial, KX2_add_KY2, phi_thres = 0.7):
    """
    Pipeline for phase unwrapping.

    Input:
        field : Complex field
        X_c : X coordinates
        Y_c : Y coordinates
        polynomial : Polynomial matrix of meshes
        KX2_add_KY2 : KX2 + KY2
        phi_thres : Threshold for phase unwrapping
    Output:
        Corrected field
    """

    #Returns the unwraped phase of the E_field
    phase_img_unwarp = phaseunwrap(np.angle(field), KX2_add_KY2)

    #Phase bakground fit
    phase_background2 = correct_phase_4order_removal(phase_img_unwarp, X_c, Y_c, polynomial, phi_thres=phi_thres)
        
    #Retrieve the phase image
    phase_image_finished = phase_img_unwarp - phase_background2

    return np.abs(field)*np.exp(1j * phase_image_finished)


def phaseunwrap(phase_image, KX2_add_KY2):
    """ 
    Phaseunwrap unwarp the 2*pi phase modulation. Works best after the polynomial background substraction. 

    Input: 
        phase_image : Wraped phase of the E-field
    Output: 
        Unwraped phase of the E-field

    """
    punw = phase_image.copy()

    php = idct2((dct2(np.cos(punw) * idct2((KX2_add_KY2) * dct2(np.sin(punw)))))* 1/(KX2_add_KY2))
    phm = idct2((dct2(np.sin(punw) * idct2((KX2_add_KY2) * dct2(np.cos(punw)))))* 1/(KX2_add_KY2))
    
    phprime = php-phm
    phprime = (phprime - phprime[9, 9]) + phase_image[9, 9] # Unsure why coordinates 9,9 are used.
    
    for _ in range(20):
        punw = punw + 2*np.pi * np.round((phprime-punw) / (2*np.pi))
        
    return punw

def correct_phase_4order_removal(phase_img, X, Y, polynomial, phi_thres = 0.7):
    """ 
    Fits a phase background to phase image, Estimates background based on a threshold phi_thres.

    Input:
        phase_img : Phase image
        X, Y : meshes
        polynomial : Predfined 4th-order polynomial.
        phi_thresh = Treshold for constructing binary image.
    Output: 
        Phase bakground fit

    """
    image = phase_img.copy()

    #Construct a simple binary image with phi_treshold. (Inverted)
    thresholded_image_inv = np.where(image > (phi_thres + np.median(image)), 1, 0)


    disk = morphology.disk(radius=6) #Some tuning of the radius  might be necessary.
    thresholded_image2 = morphology.dilation(thresholded_image_inv, footprint=disk)
    thresholded_image2_bol = np.where(thresholded_image2==0 , True, False)
    
    #Extract the x and y for the thresholded data.
    y1 = Y[thresholded_image2_bol]
    x1 = X[thresholded_image2_bol]
    
    phi_selected = image[thresholded_image2_bol]
    
    G = np.zeros((len(phi_selected), 15))
    
    G[:, 14] = 1
    G[:, 0] = x1**2
    G[:, 1] = y1*x1
    G[:, 2] = y1**2
    G[:, 3] = x1
    G[:, 4] = y1
    
    G[:, 5] = x1**3
    G[:, 6] = x1**2*y1
    G[:, 7] = x1*y1**2
    G[:, 8] = y1**3
    
    G[:, 9] = x1**4
    G[:, 10] = x1**3*y1
    G[:, 11] = x1**2*y1**2
    G[:, 12] = x1*y1**3
    G[:, 13] = y1**4
    
    dt = phi_selected
    
    # Here the coefficients to the polonomial are calculated. Note that np.linalg.lstsq(b,B)[0] is equivivalent to \ in MATLAB              
    R = np.transpose(np.linalg.cholesky(np.matmul(np.transpose(G),G)))
    
    # Equivivalent to R\(R'\(G'*dt))
    b = np.linalg.lstsq(R, 
                        np.linalg.lstsq(np.transpose(R),
                                        np.matmul(np.transpose(G), dt), rcond=None)[0], rcond=None)[0]
    
    #Here we calculate the phase background with the fitted parameters b.
    # Phase background is defined by the 4th order polynomial with the fitted parameters.
    phase_background = 0
    for i, factor in enumerate(polynomial):
        phase_background = phase_background + b[i]*factor
    phase_background = phase_background + b[14] #Adding intercept.
    
    return phase_background

def correct_phase_4order_with_background(phase, background, X, Y, polynomial):
    """ 
    Fits a phase background to phase image, based on background pixels.

    Input:
        phase : Phase image
        background : Binary matrix, 1:background, 0:not background
        X, Y : meshes
        polynomial : Predfined 4th-order polynomial.
    Output: 
        Phase bakground fit

    """
    #Extract the x and y for the background pixels.
    y1 = Y[background]
    x1 = X[background]
    
    phi_selected = phase[background]
    
    G = np.zeros((len(phi_selected), 15))
    
    G[:, 14] = 1
    G[:, 0] = x1**2
    G[:, 1] = y1*x1
    G[:, 2] = y1**2
    G[:, 3] = x1
    G[:, 4] = y1
    
    G[:, 5] = x1**3
    G[:, 6] = x1**2*y1
    G[:, 7] = x1*y1**2
    G[:, 8] = y1**3
    
    G[:, 9] = x1**4
    G[:, 10] = x1**3*y1
    G[:, 11] = x1**2*y1**2
    G[:, 12] = x1*y1**3
    G[:, 13] = y1**4
    
    dt = phi_selected
    
    # Here the coefficients to the polonomial are calculated. Note that np.linalg.lstsq(b,B)[0] is equivivalent to \ in MATLAB              
    R = np.transpose(np.linalg.cholesky(np.matmul(np.transpose(G), G)))
    
    # Equivivalent to R\(R'\(G'*dt))
    b = np.linalg.lstsq(R, 
                        np.linalg.lstsq(np.transpose(R),
                                        np.matmul(np.transpose(G), dt), rcond=None)[0], rcond=None)[0]
    
    
    # Phase background is defined by the 4th order polynomial with the fitted parameters.
    phase_background = 0
    for i, factor in enumerate(polynomial):
        phase_background = phase_background + b[i]*factor
    phase_background = phase_background + b[14] #Adding intercept.
    
    return phase_background

def get_4th_polynomial(input_shape):
    """
    Function that retrieves the 4th-order polynomial

    Input:
        input_shape : Shape of matrix
    Output :
        Polynomial matrix. 

    """
    yrc, xrc = input_shape

    xc = np.arange(-(xrc/2-1/2), (xrc/2 + 1/2), 1)
    yc = np.arange(-(yrc/2-1/2), (yrc/2 + 1/2), 1)
    Y_c, X_c = np.meshgrid(xc, yc)

    #4th order polynomial.
    polynomial = [
                    X_c**2, 
                    X_c*Y_c, 
                    Y_c**2, 
                    X_c, 
                    Y_c, 
                    X_c**3, 
                    X_c**2*Y_c, 
                    X_c*Y_c**2,
                    Y_c**3, 
                    X_c**4, 
                    X_c**3*Y_c, 
                    X_c**2*Y_c**2, 
                    X_c*Y_c**3, 
                    Y_c**4
                ]

    return polynomial

def get_G_matrix (input_shape):
    """
    Input:
        input_shape : Shape of matrix
    Output:
        Matrix to store 4-order polynominal. (Not including constant)
    """

    yrc, xrc = input_shape

    xc = np.arange(-(xrc/2-1/2), (xrc/2 + 1/2), 1)
    yc = np.arange(-(yrc/2-1/2), (yrc/2 + 1/2), 1)
    Y_c, X_c = np.meshgrid(xc, yc)

    #Vectors of equal size. x1 and y1 spatial coordinates
    x1 = 1/2 * ((X_c[1:, 1:] + X_c[:-1, :-1])).flatten(order='F')
    y1 = 1/2 * ((Y_c[1:, 1:] + Y_c[:-1, :-1])).flatten(order='F')
    
    G = np.zeros((2 * (xrc-1) * (yrc-1) , 14)) # Matrix to store 4-order polynominal. (Not including constant)
    G_end = G.shape[0]
    #(derivate w.r.t to X and Y respectively.)
    uneven_range = np.arange(1, G_end+1, 2)
    even_range = np.arange(0, G_end, 2)
    
    G[uneven_range, 4] = 1
    G[even_range, 3] = 1
    G[even_range, 1] = y1
    G[even_range, 0] = 2*x1
    G[uneven_range, 1] = x1
    G[uneven_range, 2] = 2*y1
    
    G[even_range, 5] = 3*x1**2
    G[even_range, 6] = 2*x1*y1
    G[uneven_range, 6] = x1**2
    G[uneven_range, 7] = 2*x1*y1
    G[even_range, 7] = y1**2
    G[uneven_range, 8] = 3*y1**2
    
    G[even_range, 9] = 4*x1**3
    G[even_range, 10] = 3*x1**2*y1
    G[uneven_range, 10] = x1**3
    G[even_range, 11] = 2*x1*y1**2
    G[uneven_range, 11] = 2*x1**2*y1
    G[even_range, 12] = y1**3
    G[uneven_range, 12] = 3*x1*y1**2
    G[uneven_range, 13] = 4*y1**3

    return G


def pre_calculations(
    first_frame, 
    filter_radius, 
    cropping, 
    mask_radie = [], 
    case = 'circular', 
    correct_fourier_peak = [0, 0], 
    first_phase_background = False,
    mask_out = False
    ):
    """
    When retriving the phase from a set of frames, many calculations only has to be done once. Hence precalculations can be useful to speed up computations.

    Input:
        first_frame : An 2D image
        filter_radius : How large the radius of the circular selection filter to use. For first fft
        cropping : crops away some edges. E.g. good to use 50, to avoid edge effects.
        mask_radie : creates masks with radius in list.
        case : ellipsoid or circular mask.
        correct_fourier_peak : correct fourier peak with this amount (pre analyzed that the peak is slightly shifted)
        first_phase_background : If you want to use a phase background for the first frame.
        mask_out : Create a mask that ignores certain peaks in fft image that may disturb the phase retrieval.
    Output:
        X, Y : meshes
        X_c, Y_c : meshes cropped 
        position_matrix : Used for fourier selection. Values closer to center is smaller... for tresholding.  
        G : Matrix to store 4-order polynominal. (Not including constant)
        polynomial : 4th order polynomial 
        KX, KY : meshes for wavevector
        KX2_add_KY2 : Constants in phaseunwrap function, meshes ** 2
        kx_add_ky : off-center peak position with wavevector meshes. Used for shifting 1th order component to center.
        dist_peak : coordinate**2 of off-center peak
        mask_list : list of masks for lowpass filtering.
        rad: size of the first mask
    """
    
    yr, xr  = first_frame.shape
    
    x = np.arange(-(xr/2-1/2), (xr/2 + 1/2), 1)
    y = np.arange(-(yr/2-1/2), (yr/2 + 1/2), 1)
    X, Y = np.meshgrid(x, y)
    position_matrix = np.sqrt(X**2 + Y**2) #"circle", values are smaller if closer to the center and vice verca.
    
    # Cropping the field to avoid edge effects.
    xrc, yrc = xr - cropping*2, yr - cropping*2
    
    xc = np.arange(-(xrc/2-1/2), (xrc/2 + 1/2), 1)
    yc = np.arange(-(yrc/2-1/2), (yrc/2 + 1/2), 1)
    Y_c, X_c = np.meshgrid(xc, yc)
    
    #4th order polynomial.
    polynomial = [
        X_c**2, 
        X_c*Y_c, 
        Y_c**2, 
        X_c, 
        Y_c, 
        X_c**3, 
        X_c**2*Y_c, 
        X_c*Y_c**2, 
        Y_c**3, 
        X_c**4, 
        X_c**3*Y_c, 
        X_c**2*Y_c**2, 
        X_c*Y_c**3, 
        Y_c**4
        ]
    
    #Vectors of equal size. x1 and y1 spatial coordinates
    x1 = 1/2 * ((X_c[1:, 1:] + X_c[:-1, :-1])).flatten(order='F')
    y1 = 1/2 * ((Y_c[1:, 1:] + Y_c[:-1, :-1])).flatten(order='F')
    
    G = np.zeros((2 * (xrc-1) * (yrc-1) , 14)) # Matrix to store 4-order polynominal. (Not including constant)
    G_end = G.shape[0]
    #(derivate w.r.t to X and Y respectively.)
    uneven_range = np.arange(1, G_end+1, 2)
    even_range = np.arange(0, G_end, 2)
    
    G[uneven_range, 4] = 1
    G[even_range, 3] = 1
    G[even_range, 1] = y1
    G[even_range, 0] = 2*x1
    G[uneven_range, 1] = x1
    G[uneven_range, 2] = 2*y1
    
    G[even_range, 5] = 3*x1**2
    G[even_range, 6] = 2*x1*y1
    G[uneven_range, 6] = x1**2
    G[uneven_range, 7] = 2*x1*y1
    G[even_range, 7] = y1**2
    G[uneven_range, 8] = 3*y1**2
    
    G[even_range, 9] = 4*x1**3
    G[even_range, 10] = 3*x1**2*y1
    G[uneven_range, 10] = x1**3
    G[even_range, 11] = 2*x1*y1**2
    G[uneven_range, 11] = 2*x1**2*y1
    G[even_range, 12] = y1**3
    G[uneven_range, 12] = 3*x1*y1**2
    G[uneven_range, 13] = 4*y1**3
    
    #### Constants in phaseunwrap function.
    KY_, KX_ = np.meshgrid(np.arange(1, xrc+1,1), np.arange(1, yrc+1, 1))
    KX2_add_KY2 = KX_**2+KY_**2
    
    #### kx and ky is the wave vector that defines the direction of propagation. Used to calculate the fourier shift.
    kx = np.linspace(-np.pi, np.pi, xr) 
    ky = np.linspace(-np.pi, np.pi, yr)
    KX, KY = np.meshgrid(kx, ky)
    
    ##### The peak coordinates in the fourier space are the same for all frames (should be very similar atleast.)
    fftImage = np.fft.fft2(first_frame) #Compute the 2-dimensional discrete Fourier Transform
    fftImage = np.fft.fftshift(fftImage) #Shift the zero-frequency component to the center of the spectrum.
    
    yr, xr = fftImage.shape 
    if not isinstance(filter_radius, (int, np.uint8)):
        filter_radius = int(np.min([xr, yr]) / 7)

    fftImage = np.where(position_matrix < filter_radius, 0, fftImage) #Set values within filter_radius to 0
    fftImage = np.where(X < -5, 0, fftImage) #Set "left" values to 0 
    fftImage = np.where(np.abs(Y) < 5, 0, fftImage) #Set "fourier boundary" to 0. 

    #Find max with some minor gaussian convolutions
    imag_c = scipy.ndimage.gaussian_filter(fftImage.imag, sigma = 3)
    real_c = scipy.ndimage.gaussian_filter(fftImage.real, sigma = 3)
    idx_max_real = np.unravel_index(np.argmax(real_c, axis=None), fftImage.shape)
    idx_max_imag = np.unravel_index(np.argmax(imag_c, axis=None), fftImage.shape)
    
    idx_max = (int((idx_max_real[0] + idx_max_imag[0])/2), int((idx_max_real[1] + idx_max_imag[1])/2))   
    idx_max = (idx_max[0]+correct_fourier_peak[0], idx_max[1]+correct_fourier_peak[1])
    idx_max = (idx_max[0], idx_max[1])

    x_pos = X[idx_max] #In X
    y_pos = Y[idx_max] #In Y
    dist_peak = np.sqrt(x_pos**2 + y_pos**2)

    kx_pos = KX[idx_max]
    ky_pos = KY[idx_max]
    kx_add_ky = kx_pos*X+ky_pos*Y

    #Precalculate masks that will be used multiple times.
    mask_list = []
    if len(mask_radie) > 0:
        for i, rad in enumerate(mask_radie):
            #i<2 are masks before cropping.
            if i < 2:
                if case == 'ellipse': mask_list.append(create_ellipse_mask(yr, xr, percent = rad / yr))
                elif case == 'circular': mask_list.append(create_circular_mask(yr, xr, radius = rad))
            else:
                if case == 'ellipse': mask_list.append(create_ellipse_mask(yrc, xrc, percent = rad / yr))
                elif case == 'circular': mask_list.append(create_circular_mask(yrc, xrc, radius = rad))
        rad = mask_radie[0]
    #We estimate the size of the fourier filter. For first filtering step only.
    else:
        rad = int(round(np.max([xr, yr]) / 6))
        if case == 'ellipse': mask_list.append(create_ellipse_mask(yr, xr, percent = rad / yr))
        elif case == 'circular': mask_list.append(create_circular_mask(yr, xr, radius = rad))

    #FFT image shifted to center and masked out
    fftImage2 = np.fft.fftshift(
            np.fft.fft2(
            (first_frame - np.mean(first_frame)) * np.exp(1j*(kx_add_ky)))
            ) * mask_list[0]

    if mask_out:
        new_mask = mask_out_pipeline(
            fftImage2, 
            mask_list[0], 
            min_distance = 25, 
            sigma = 2.75, 
            max_peaks = 6, 
            min_distance_from_center = 100, 
            mask_out_size = 25, 
            mask_out_case = case)

        #Update mask list.
        mask_list[0] = new_mask

    #Here we calculate the phase background for the first image.
    if first_phase_background:
        if len(mask_list)>1:
            phase_img = phase_frequencefilter(fftImage2, mask = mask_list[1] , is_field = False, crop = cropping)
        else:
            if cropping > 0: 
                phase_img = np.angle(np.fft.ifft2(np.fft.fftshift(fftImage2))[cropping:-cropping, cropping:-cropping])
            else: 
                phase_img = np.angle(np.fft.ifft2(np.fft.fftshift(fftImage2)))
        # Get the phase background from phase image.
        phase_background = correct_phase_4order(phase_img, G, polynomial)
    else:
        phase_background = []

    return X, Y, X_c, Y_c, position_matrix, G, polynomial, KX, KY, KX2_add_KY2, kx_add_ky, dist_peak, mask_list, phase_background, rad

def get_shifted_fft(frame, filter_radius=200, correct_fourier_peak = [0, 0]):
    """
    Return fourier image that is shifted by some pixels.
    
    """

    yr, xr  = frame.shape
    
    x = np.arange(-(xr/2-1/2), (xr/2 + 1/2), 1)
    y = np.arange(-(yr/2-1/2), (yr/2 + 1/2), 1)
    X, Y = np.meshgrid(x, y)

    #### kx and ky is the wave vector that defines the direction of propagation. Used to calculate the fourier shift.
    kx = np.linspace(-np.pi, np.pi, xr) 
    ky = np.linspace(-np.pi, np.pi, yr)
    KX, KY = np.meshgrid(kx, ky)

    position_matrix = np.sqrt(X**2 + Y**2) #"circle", values are smaller if closer to the center and vice verca.

    fftImage = np.fft.fft2(frame) #Compute the 2-dimensional discrete Fourier Transform
    fftImage = np.fft.fftshift(fftImage) #Shift the zero-frequency component to the center of the spectrum.

    #If not filter radius inputted. Estimate it somewhat. 
    if not isinstance(filter_radius, int):
        filter_radius = int(np.min([xr, yr]) / 6) 

    fftImage = np.where(position_matrix < filter_radius, 0, fftImage) #Set values within filter_radius to 0
    fftImage = np.where(X < -5, 0, fftImage) #Set "left" values to 0 
    fftImage = np.where(np.abs(Y) < 5, 0, fftImage) #Set "fourier boundary" to 0. 
    
    #Find max with some minor gaussian convolutions
    imag_c = scipy.ndimage.gaussian_filter(fftImage.imag, sigma = 3)
    real_c = scipy.ndimage.gaussian_filter(fftImage.real, sigma = 3)
    idx_max_real = np.unravel_index(np.argmax(real_c, axis=None), fftImage.shape)
    idx_max_imag = np.unravel_index(np.argmax(imag_c, axis=None), fftImage.shape)
    
    idx_max = (int((idx_max_real[0] + idx_max_imag[0])/2), int((idx_max_real[1] + idx_max_imag[1])/2))   
    
    idx_max = (idx_max[0]+correct_fourier_peak[0], idx_max[1]+correct_fourier_peak[1])
    #idx_max = (idx_max[0]+5, idx_max[1]-25)
    #gauss_fft = scipy.ndimage.gaussian_filter(np.log(np.abs(fftImage)), sigma = 5)
    #idx_max = np.unravel_index(np.argmax(gauss_fft, axis=None), fftImage.shape)

    #x_pos = X[idx_max] #In X
    #y_pos = Y[idx_max] #In Y
    #dist_peak = np.sqrt(x_pos**2 + y_pos**2)

    kx_pos = KX[idx_max]
    ky_pos = KY[idx_max]
    kx_add_ky = kx_pos*X+ky_pos*Y

    ###Center on one of the off-center peaks
    img = np.array(frame, dtype = np.float32) 
    img = img - np.mean(img) #img = img - np.mean(img)
    
    #Compute the 2-dimensional discrete Fourier Transform with offset image.
    fftImage = np.fft.fft2(img * np.exp(1j*(kx_add_ky)))

    #shifted fourier image centered on peak values in x and y. 
    fftImage = np.fft.fftshift(fftImage)

    return fftImage


def imgtofield_simple(img, 
               G, 
               polynomial, 
               KX2_add_KY2, 
               kx_add_ky,
               X_c,
               Y_c,
               position_matrix,
               dist_peak,
               cropping=50,
               mask_f_case = 'sinc',
               ):
    """
    Function that takes in a set of precalculated matrices and scalars to reconstruct an optical field from the interference pattern in image.
    Works well for "easy" holographic data.

    Input:
        img : An 2D image
        G : Matrix to store 4-order polynominal. (Not including constant)
        polynomial : 4th order polynomial 
        KX2_add_KY2 : precalculated matrix for phase unwraping
        kx_add_ky : offset for shifting image to one of the off-center peaks.
        Y_c : cropped Y mesh
        X_c : cropped X mesh
        position_matrix : Used for fourier selection. Values closer to center is smaller... for tresholding. 
        dist_peak = coordinate**2 of off-center peak
        mask_f_case : weight fourier image with function. 'sinc', 'jinc, or no weighting.
        radius_fourier_selection : Radius of the circular selection filter to use.
        cropping : crops away some edges. E.g. good to use 50, to avoid edge effects. Important to keep same as in precalculations.
    Output:
        Complex valued optical field.

    """

    #Make image float.
    img = np.array(img, dtype = np.float32) 

    #Compute the 2-dimensional fourier transform with offset kx_add_ky.
    fftImage = np.fft.fft2(img * np.exp(1j*(kx_add_ky)))

    #shifted fourier image centered on peak values in x and y. 
    fftImage = np.fft.fftshift(fftImage)
    
    #Selection fourier filter.
    selection_filter = position_matrix > dist_peak / 3 

    #Sets values outside the defined circle to zero. Ie. take out the information for this peak.
    fftImage2 = np.where(selection_filter, 0, fftImage)

    #Scale fftimage with sinc function
    if mask_f_case == 'sinc':
        fftImage2 = fftImage2 * np.sinc(selection_filter)
    elif mask_f_case == 'jinc':
        fftImage2 = fftImage2 * jinc(selection_filter)

    #Retrieve optical field.
    E_field = np.fft.ifft2(np.fft.fftshift(fftImage2)) 
    
    #Crop optical field to avoid edge effects.
    if cropping > 0:
        E_field_cropped = E_field[cropping:-cropping, cropping:-cropping]
    else:
        E_field_cropped = E_field
    
    #Get phase image.
    phase_img  = np.angle(E_field_cropped) 
    
    # Get the phase background from phase image.
    phase_background = correct_phase_4order(phase_img, G, polynomial)
    
    #Correct E_field with the phase_background
    E_field_corr = E_field_cropped * np.exp(- 1j * phase_background)

    #Get phase image.
    phase_img2 = np.angle(E_field_corr)
    
    #Correct E_field again
    E_field_corr2 = E_field_corr * np.exp(- 1j * np.median(phase_img2 + np.pi - 1))

    #Get phase image.
    phase_img3 = np.angle(E_field_corr2)
    
    #Unwrap the phase. Quite slow, but improves reconstruction.
    phase_img_unwarp = phaseunwrap(phase_img3, KX2_add_KY2)

    #Phase bakground fit for unwraped phase
    phase_background2 = correct_phase_4order_removal(phase_img_unwarp, X_c, Y_c, polynomial, phi_thres = 0.5) 

    #Substract background to retrieve final phase_image
    phase_image_finished = phase_img_unwarp - phase_background2 
        
    #Final optical field.    
    E_field_corr3 = np.abs(E_field_corr2)*np.exp(1j * phase_image_finished)
        
    return E_field_corr3

def phase_frequencefilter(field, mask, is_field = True, crop = 0):
    """
    Lowpass filter the image with mask defined in inpu.

    Input:
        field : Complex valued field.
        mask : Mask (binary 2D image)
        is_field : if input is a field, or if needs to be forier transformed before.
        crop : if we shall crop the ouput slightly
    Output:
        phase_img : phase of optical field.
    """
    if is_field:
        freq = np.fft.fftshift(np.fft.fft2(field))
    else:
        freq = field
        
    #construct low-pass mask
    freq_low = freq * mask
    
    E_field = np.fft.ifft2(np.fft.fftshift(freq_low)) #Shift the zero-frequency component to the center of the spectrum. and compute inverse fft
    
    if crop > 0:
        phase_img = np.angle(E_field[crop:-crop, crop:-crop])
    else:
        phase_img = np.angle(E_field)
    
    return phase_img

def correct_phase_manually(field, phase_img, margin_shift = 2.5, input_phase = True):
    """
    Function for manually doing phase unwrapping (or similar to phase unwrap atleast)

    Input:
        field : 2D complex valued array
        phase_img : phase of field
        margin_shift : constant we use to shift phase
        input_phase : if we input a phase image or not.
    Ouput:
        Corrected phase.

    """
    shift_from_median = np.pi - margin_shift
    
    #If phase inputed, use that
    if input_phase:
        phase_shift = -np.median(phase_img) - shift_from_median
        phase0 = phase_img+phase_shift
        phase1 = phase_img+phase_shift-4
    else:    
        field_test = field
        phase_shift = -np.median(np.angle(field_test)) - shift_from_median
        phase0 = np.angle(field_test * np.exp(1j * phase_shift))
        phase1 = np.angle(field_test * np.exp(1j * (phase_shift - 4)))
    
    # Create mask where phase larger than 1.5
    filt_warp = ndimage.gaussian_filter(phase0>1.5, sigma = 0.25)
    
    #Compute phase
    phase = phase0 * (filt_warp<1) + (phase1 + 4) * filt_warp + shift_from_median
    
    phase = phase - np.median(phase) - 0.3
    
    return phase

def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a circular mask.

    Input:
        h : height
        w : width
        center : Define center of image. If None -> middle is used.
        radius : radius of circle.
    Output:
        Circular mask.

    """
    
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
        
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def create_ellipse_mask(h, w, center = None, radius_h = None, radius_w = None, percent = 0.05):
    """
    Creates a ellipsoid mask.

    Input:
        h : height
        w : width
        center : Define center of image. If None -> middle is used
        radius_h : Radius in height
        radius_w : radius in width
        percent : if radius_h or radius_w is not defined use this percentage factor instead.
    Output:
        Ellipsoid mask.

    """

    import cv2 #Implementation is done in cv2

    if center is None:
        center_w, center_h = int(w/2), int(h/2)
    else:
        center_w, center_h = center[0], center[1]

    if radius_h is None and radius_w is None: 
    
        if percent is not None:
            radius_w, radius_h =  int(percent * w), int(percent * h)
        else:    
            radius_w, radius_h =  int(0.25 * w/2), int(0.25 * h/2) #Ellipsoid of this size. To get some output
    
    img = np.zeros((h, w))
    mask = cv2.ellipse(img, (center_w, center_h), (radius_w, radius_h), 0, 0, 360, 255, -1)

    return np.where(mask > 0, 1, 0)

def mask_out_pipeline(
    fftimage, 
    mask_original, 
    min_distance = 50, 
    sigma = 3, 
    max_peaks = 6, 
    min_distance_from_center = 50, 
    mask_out_size = 30, 
    mask_out_case = 'ellipse'):

    """
    Mask out peaks in fftimage.
    """

    #Import libraries-Used for detecting the peaks
    from skimage.feature import peak_local_max

    intensity = np.abs(fftimage)

    #Normalize intensity within 0 - 1
    intensity = intensity / np.max(intensity)

    #Add a small delta
    intensity = intensity + 1e-9

    #Do gaussian blur
    intensity = scipy.ndimage.gaussian_filter(intensity, sigma=sigma)

    #Normalize intensity within 0 - 1
    intensity = intensity / np.max(intensity)

    #Find local maximas
    local_maxi = peak_local_max(intensity, min_distance=min_distance)

    #Find distance from center
    distance_from_center = np.sqrt((local_maxi[:,0] - fftimage.shape[0]/2)**2 + (local_maxi[:,1] - fftimage.shape[1]/2)**2)

    #Extract the local maximas that are far enough from center
    local_maxi = local_maxi[distance_from_center > min_distance_from_center]

    #Max peaks
    peaks = local_maxi[:max_peaks]

    #Mask out the peaks
    mask = np.zeros_like(mask_original)

    for peak in peaks:
        
        if mask_out_case == 'ellipse':
            m = create_ellipse_mask(mask_original.shape[0], mask_original.shape[1], center = [peak[1], peak[0]], percent = mask_out_size / mask_original.shape[1])

        elif mask_out_case == 'circle':
            m = create_circular_mask(mask_original.shape[0], mask_original.shape[1], center = [peak[1], peak[0]], radius = mask_out_size)
        
        mask = mask + m

    mask[mask>0] = -1

    new_mask = np.array(mask_original + mask, dtype = np.int8)

    new_mask = np.where(new_mask == -1, 0, new_mask)

    return new_mask

def black_frame(img):
    """
    Function checks if the current frame is completely black.

    Input:
        img : 2D image
    Output:
        boolean
    """
    
    if np.sum(img==0) < (img.shape[0] * img.shape[1]):
        return False 
    else:
        return True 

def dct(y):
    """
    Type-II discrete cosine transform (DCT) of real data y

    Input:
        y : Real arrayed image, e.g. phase image.
    
    """
    N = len(y)
    y2 = np.empty(2*N, float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = np.fft.rfft(y2)
    phi = np.exp(-1j*np.pi*np.arange(N)/(2*N))
    return np.real(phi*c[:N])    

def dct2(y):
    """
    2D DCT of 2D real array y

    Input:
        y : Real arrayed image, e.g. phase image.

    """
    M, N = y.shape
    a = np.empty([M,N], float)
    b = np.empty([M,N], float)

    for i in range(M):
        a[i,:] = dct(y[i,:])
    for j in range(N):
        b[:,j] = dct(a[:,j])

    return b    

def idct(a):
    """
    Type-II inverse DCT of a

    Input:
        a : Real arrayed image, e.g. phase image.

    """
    N = len(a)
    c = np.empty(N+1, complex)

    phi = np.exp(1j*np.pi*np.arange(N)/(2*N))
    c[:N] = phi*a
    c[N] = 0.0
    return np.fft.irfft(c)[:N]    

def idct2(b):
    """
    2D inverse DCT of real array

    Input:
        b : Real arrayed image, e.g. phase image.

    """
    M, N = b.shape
    a = np.empty([M,N], float)
    y = np.empty([M,N], float)

    for i in range(M):
        a[i,:] = idct(b[i,:])
    for j in range(N):
        y[:,j] = idct(a[:,j])
    return y

def scale(x, out_range=(-1, 1)):
    """
    Function for scaling values within a certain range.

    Input:
        x : Image
        out_range : Values to scale image to be within.
    Output:
        Transformed Image 
    """
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def jinc(x): 
    """
    Jinc function of input x.

    Input:
        x : Image
    Output:
        Transformed Image
    """
    return  scipy.special.j1(x) / x


