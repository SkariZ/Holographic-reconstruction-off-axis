
"""
Script containing functions for propagating an optical field.

Created on Wed Mar 03 09:14:39 2022

@author: Fredrik Skärberg // fredrik.skarberg@physics.gu.se

"""

import cupy as cp
from cupyx.scipy import ndimage

from scipy import ndimage as ndi

def precalc_Tz (k, zv, K, C):
    """
    Computes the T_z matrix used for propagating an optical field.

    Input:
        k : Wavevector
        zv : List of z propagation steps
        K : Transformation matrix
        C : Circular disk
    Output:
        z-propagator matrix

    """

    return [C*cp.fft.fftshift(cp.exp(k * 1j*z*(K-1))) for z in zv]

def precalc(field, wavelength=0.532):
    """
    Precalculate some constants for propagating field for faster computations.

    Input:
        field : Complex valued optical field.
        wavelength : in um
    Output:
        K : Transformation matrix
        C : Circular disk

    """
    
    k = 2 * cp.pi / wavelength * 1.33 # Wavevector
    
    yr, xr = field.real.shape

    x = 2 * cp.pi/(.114) * cp.arange(-(xr/2-1/2), (xr/2 + 1/2), 1)/xr
    y = 2 * cp.pi/(.114) * cp.arange(-(yr/2-1/2), (yr/2 + 1/2), 1)/yr

    KXk, KYk = cp.meshgrid(x, y)
    K = cp.real(cp.sqrt(cp.array(1 -(KXk/k)**2 - (KYk/k)**2 , dtype = cp.complex64)))
    
    #Create a circular disk here.
    C = cp.fft.fftshift(((KXk/k)**2 + (KYk/k)**2 < 1)*1.0)

    return x, y, K, C 

def refocus_field_z(field, z_prop, padding = 0, wavelength = 0.532):
    """
    Function for refocusing field.

    Input:
        field : Complex valued optical field.
        z_prop : float or integer of amount of z propagation.
        padding : Pad image by this amount in all directions
    Output: 
         Refocused field

    """

    if not cp.iscomplexobj(field):
        field = field[..., 0] + 1j*field[..., 1]
        
    if padding > 0:
        field = cp.pad(field, ((padding, padding), (padding, padding)), mode = 'reflect')

    k = 2 * cp.pi / wavelength * 1.33 # Wavevector
    _, _ , K, C = precalc(field, wavelength)
    
    z = [z_prop]
    
    #Get matrix for propagation.
    Tz = precalc_Tz(k, z, K, C)    
    Tz = cp.asarray(Tz)

    #Fourier transform of field
    f1 = cp.fft.fft2(field)
    
    #Propagate f1 by z_prop
    refocused = cp.fft.ifft2(Tz*f1)
    
    if padding > 0:
        refocused = refocused[:, padding:-padding, padding:-padding]

    return cp.squeeze(refocused)


def refocus_field(field, steps=51, interval = [-10, 10], padding = 0, wavelength = 0.532):
    """
    Function for refocusing field.

    Input:
        field : Complex valued optical field.
        Steps : N progation steps, equally ditributed in interval.
        Interval : Refocusing interval
        padding : Pad image by this amount in all directions

    Output: 
        A stack of propagated_fields

    """
    if not cp.iscomplexobj(field):
        field = field[..., 0] + 1j*field[..., 1]

    if padding > 0:
        field = cp.pad(field, ((padding, padding), (padding, padding)), mode = 'reflect')

    k = 2 * cp.pi / wavelength * 1.33 # Wavevector
    _, _, K, C = precalc(field)
    
    zv = cp.linspace(interval[0], interval[1], steps)
    
    #Get matrix for propagation.
    Tz = precalc_Tz(k, zv, K, C)    
    
    #Fourier transform
    f1 = cp.fft.fft2(field)
    
    #Stack of different refocused images.
    refocused =  cp.array([cp.fft.ifft2(Tz[i]*f1) for i in range(steps)], dtype = cp.complex64)
    
    if padding > 0:
        refocused = refocused[:, padding:-padding, padding:-padding]

    return refocused
    
def find_focus_field(field, steps=51, interval = [-10, 10], m = 'fft', padding=0, ma=0, use_max_real = False, bbox = [], wavelength = 0.532):
    """
    Find focus of optical field.

    Input:
        field : Complex valued optical field.
        Steps : N progation steps, equally ditributed in interval.
        Interval : Refocusing interval
        m : Evaluation criterion, 'abs', 'sobel', or 'adjescent', 'tamura'
        padding : Pad image by this amount in all directions
        use_max_real : simply take the max value of the real part of the optical field and extract an roi around that point.
        bbox : evaluate the focus in the region of the bounding box.

    Output: 
        z_prop : float optimal z propagation

    """

    if not cp.iscomplexobj(field):
        field = field[..., 0] + 1j*field[..., 1]

    #Interval to propagate within
    zv = cp.linspace(interval[0], interval[1], steps)
    
    #Predefined bbox
    if len(bbox)==4 and use_max_real == False:
        field = field[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        
    elif use_max_real==True:
        idx_max = cp.unravel_index(cp.argmax(field.real, axis=None), field.real.shape)
        padsize = 64
        
        #Use an roi where max is found
        if idx_max[0]-padsize > 0 and idx_max[0]+padsize<field.shape[0] and idx_max[1]-padsize > 0 and idx_max[1]+padsize<field.shape[1]:
            field = field[idx_max[0]-padsize : idx_max[0]+padsize, idx_max[1]-padsize : idx_max[1]+padsize]


    field = refocus_field(field, 
                        steps=steps, 
                        interval=interval,
                        padding=padding,
                        wavelength=wavelength
                        )
       
                        
    #Some ways of finding criterions.
    if m == 'fft':
        criterion = [-(cp.std((cp.fft.fft2(im)).real) + cp.std((cp.fft.fft2(im)).imag)) for im in field]    
    elif m == 'abs':
        criterion = [cp.std(cp.abs(im)) for im in field]
    elif m == 'maxabs':
        criterion = [cp.max(im.real) + cp.max(im.imag) for im in field]
    elif m == 'maxabs2':
        criterion = [cp.max(cp.abs(im.real)) + cp.max(cp.abs(im.imag)) for im in field]
    elif m == 'sobel':
        criterion = [-(cp.std(ndimage.sobel(im.real)) + cp.std(ndimage.sobel(im.imag))) for im in field]
    elif m == 'adjescent':
        n_rows = int(0.5 * field.shape[0])
        criterion = [adjescent_pixels(im.imag, n_rows) for im in field]
    elif m == 'tamura':
        ### OBS quite slow for big images...
        criterion = [Tamura_coefficient(SoG(im)) for im in field]
    elif m =='classic':
        abssqim = [cp.abs(im)**2 for im in field]
        criterion = [cp.sum(cp.abs(ab - cp.mean(ab))**2) for ab in abssqim]
    elif m =='local_maximas':
        criterion = [local_maximas(im) for im in field]

    criterion = cp.array(criterion)

    if ma>1 and len(criterion)>ma:
        criterion = moving_average(criterion, ma)

    #idx of max 
    idxmax = cp.argmax(criterion)
    #How much propagation in z
    z_focus = zv[idxmax]
    
    return z_focus, criterion

def find_focus_field_stack(field,  m = 'fft', padmin = 6, ma=0):
    """
    Find focus of optical field and return the "most" focused image in stack.

    Input:
        field : Complex valued optical field as a stack.
        m : Evaluation criterion
        padmin : crop image slightly.
        ma : moving average of metric array
    Output: 
        z_prop : Return focused image

    """
    F = field.copy()

    #Make ROI smaller
    if padmin < int(field.shape[1] / 2):
        field = field[:, padmin:-padmin, padmin:-padmin]

    #Make it complex if not complex
    if not cp.iscomplexobj(field):
        field = field[..., 0] + 1j*field[..., 1]

    #Calculate criterions.
    if m == 'fft':
        criterion = [-(cp.std((cp.fft.fft2(im)).real) + cp.std((cp.fft.fft2(im)).imag)) for im in field] 
    elif m == 'abs':
        criterion = [-cp.std(cp.abs(im)) for im in field]
    elif m == 'maxabs':
        criterion = [cp.max(im.real) + cp.max(im.imag) for im in field]
    elif m == 'maxabs2':
        criterion = [cp.max(cp.abs(im.real)) + cp.max(cp.abs(im.imag)) for im in field]
    elif m == 'sobel':
        criterion = [-(cp.std(ndimage.sobel(im.real)) + cp.std(ndimage.sobel(im.imag))) for im in field]
    elif m == 'adjescent':
        n_rows = int(0.5 * field.shape[0])
        criterion = [adjescent_pixels(im, n_rows) for im in field]
    elif m == 'tamura':
        ### OBS quite slow for big images...
        criterion = [Tamura_coefficient(SoG(im)) for im in field]
    elif m =='classic':
        abssqim = [cp.abs(im)**2 for im in field]
        criterion = [cp.sum(cp.abs(ab - cp.median(ab))**2) for ab in abssqim]
    elif m =='local_maximas':
        criterion = [local_maximas(im) for im in field]

    if ma>1 and len(criterion)>ma:
        criterion = moving_average(criterion, ma)

    #idx of max 
    idxmax = cp.argmax(criterion)
    
    #Return the max.
    focused_field = F[idxmax]
    
    return focused_field

def local_maximas(im):
    """
    Find local maximas in image.
    """
    im = cp.abs(im)
    im = im - cp.min(im)
    im = im / cp.max(im)
    im = im.get()
    #Convert im to 0-255
    im = im * 255
    im = im.astype('uint8')
    #Find local maximas in image with scipy ndimage
    import cv2
    maxValue = 255
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C#cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresholdType = cv2.THRESH_BINARY#cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
    blockSize = 7 #odd number like 3,5,7,9,11
    C = -3 # constant to be subtracted
    im_thresholded = cv2.adaptiveThreshold(im, maxValue, adaptiveMethod, thresholdType, blockSize, C) 
    _, particle_count = ndi.measurements.label(im_thresholded)
    return cp.asarray(particle_count)

def Tamura_coefficient(vec):

    """
    Compute the Tamura coefficient.
    """
    return cp.sqrt(cp.std(vec) / cp.mean(vec))

def SoG(field):

    """
    Sparsity of the Gradient (SoG)

    Use this for a small region, otherwise it is slow.

    Input:
        Complex valued optical field.
    Output:
        Sparsity of the gradient, to be used to calculate in focus.
    """
    grad_x = cp.abs(field[1:, 1:] - field[1:, :-1])**2
    grad_y = cp.abs(field[1:, 1:] - field[:-1, 1:])**2

    res = cp.sqrt(grad_x + grad_y)

    #res = []
    #for i in range(1, row):
    #    for j in range(1, col):

     #       res.append(
     #           cp.sqrt(
     #               cp.abs(field[i, j] - field[i, j-1])**2 + \
     #               cp.abs(field[i, j] - field[i-1, j])**2
     #               )
     #       )
    
    return res

def adjescent_pixels(image, n_rows):
    """
    Compute the sum of absolute value between pixels in image.

    Input:
        Image : Complex valued image of optical field. (must not be complex, works anyways...)
        n_rows : How many rows to consider
    """
    abssum = 0
    for i in range(n_rows-1):
        abssum += cp.sum(cp.abs(image[i, :].real - image[i+1, :].real))
        
    return float(abssum / n_rows) 

def moving_average(x, w):
    return cp.convolve(x, cp.ones(w), 'valid') / w 