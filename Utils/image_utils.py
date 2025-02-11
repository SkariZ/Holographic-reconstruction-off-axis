import cv2

import scipy
from scipy import ndimage
import cupy as cp
import glob
import re

import matplotlib.pyplot as plt
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text) ]

#Save frame in native resolution. Can change colormap if necessary.
def save_frame(frame, folder, name, cmap = 'gray', annotate = 'False', annotatename='', dpi = 300):

    frame = frame.get()

    plt.imsave(f"{folder}/{name}.png", frame, cmap = cmap)

    if annotate:
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap = cmap)
        ax.annotate(
            annotatename, 
            xy=(0.015, 0.985),
            xycoords='axes fraction', 
            fontsize=14, 
            horizontalalignment='left', 
            verticalalignment='top',
            color = 'white'
            )
        ax.axis('off')
        fig.savefig(f"{folder}/{name}.png", bbox_inches="tight", pad_inches = 0, dpi = dpi)
        plt.close(fig)


def cropping_image(image, h, w, corner):
    """
    Crops the image
    """
    
    hi, wi = image.shape[:2]
    if hi<=h or wi<=w:
        raise Exception("Cropping size larger than actual image size.")
    
    #if wi == hi:#If we have a square image we can resize without loss of quality
    #    image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
   #Crop out the "corner"
    if corner == 1:
        image = image[:h, :w] #Top left
    elif corner == 2:
        image = image[:h, -w:] #Top right
    elif corner == 3:
        image = image[-h:, :w] #Bottom left
    elif corner == 4:
        image = image[-h:, -w:] #Bottom right
    #elif wi == hi:
    #    image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

    return image
    
def first_frame(video, height, width, H, W, corner, edges=False, index=1):
    """
    Returns the first frame of the video. Used for precalculations.
    """
    video.set(1, index); # Where index is the frame you want
    _, frame = video.read() # Read the frame
    image = frame[:, :, 0]
    
    if edges:
        image = clip_image(image)
        height, width = image.shape[:2]
        
    if H<height and H != 1 or W>width and W !=1:
        image = cropping_image(image, H, W, corner=corner)
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

def downsample2d(inputArray, kernelSize):
    average_kernel = cp.ones((kernelSize, kernelSize)).get()
    inputArray = inputArray.get()

    blurred_array = scipy.signal.convolve2d(inputArray, average_kernel, mode='same')
    downsampled_array = blurred_array[::kernelSize,::kernelSize]
    return cp.asarray(downsampled_array)

def correctfield(field, n_iter = 5):
    """
    Correct field
    """
    import numpy as np

    field = (field.get()).astype(np.complex64)

    f_new = field.copy()

    #Normalize with mean of absolute value.
    f_new = f_new / np.mean(np.abs(f_new))

    for _ in range(n_iter):
        f_new = f_new * np.exp(-1j * np.median(np.angle(f_new)))

    return cp.asarray(f_new)

def correctfield_sign(field, pos = True):
    """
    Force mean of the real part to be positive.
    """
    import numpy as np

    field = (field.get()).astype(np.complex64)

    if pos:
        for i, f in enumerate(field):
            if np.mean(np.real(f)) < 0:
                field[i] = -f
    else:
        for i, f in enumerate(field):
            if np.mean(np.real(f)) > 0:
                field[i] = -f
    return cp.asarray(field)

def correctfield_phase0(field):
    """
    Normalize each frame so that the mean of the phase is zero.
    """
    import numpy as np

    field = (field.get()).astype(np.complex64)
    f_new = field.copy()

    bg_box = [0, 0, 40, 40]

    for i, f in enumerate(f_new):
        f_new[i] = f * np.exp(-1j * np.mean(
            np.angle(
            f[bg_box[0]:bg_box[0]+bg_box[2], bg_box[1]:bg_box[1]+bg_box[3]]
            )
            ))

    return cp.asarray(f_new)

def background_segmentation(data, n_frames = 0, gaussian_sigma = 8, classes = 2, remove_size = 100, disk = 5, case = 'phase', ndimage_open = True):
    """
    Function that segments an image as cell or no cell.
    Works best with imaginary part of the optical field or the phase image.
    """

    import numpy as np
    import skimage
    import skimage.morphology as morphology
    
    data = (data.get()).astype(np.complex64)

    if case=='phase':
        data_seg = np.angle(data)
    elif case == 'imag':
        data_seg = data.imag
    elif case == 'abs':
        data_seg = np.abs(data)

    maskvec = np.zeros((data.shape), dtype = np.uint8)

    #If n_frames = 0, we use all frames.
    if n_frames == 0:
        n_frames = len(data_seg)

    maskvec = maskvec[:n_frames]

    for i in range(n_frames):
        image = data_seg[i]

        if np.sum(image) == 0 or np.sum(~np.isnan(image))==0:
            image = np.random.normal(0, 3, (image.shape))

        if gaussian_sigma > 0:
            im = skimage.filters.gaussian(image, sigma = gaussian_sigma)

        #Multi-Otsu threshold
        T = skimage.filters.threshold_multiotsu(im, classes = classes) 
        regions = np.digitize(im, bins=T)

        #Removing small holes and small objects etc.
        mask = morphology.remove_small_holes(
            morphology.remove_small_objects(
                np.array(regions, dtype = bool), remove_size),
            remove_size)

        if disk > 0:
            mask = morphology.opening(mask, morphology.disk(disk))

        mask = np.where(mask == True, 1, 0)

        if ndimage_open:
            mask = scipy.ndimage.binary_fill_holes(mask)

        maskvec[i] = mask

    maskvec = np.array(maskvec, dtype = np.uint8)
    
    return maskvec

def background_normalize(field, background):
    """
    Normalize each frame so that the mean of the phase is zero.
    """
    import numpy as np

    field = (field.get()).astype(np.complex64)
    f_new = field.copy()

    #invert background so 1 is background
    background = np.where(background == 0, 1, 0)

    #Take mean of background pixels
    background = np.mean(background, axis = 0)

    #Make background binary
    background = np.where(background > 0.6, 1, 0)
    
    #Check so that background does not contain all zeros
    if np.sum(background) == 0:
        background = np.ones(background.shape)

    #Vectorize
    background = background.flatten()

    for i, f in enumerate(f_new):
        
        #Take background pixels
        bg_f = f.flatten()[background]

        f_new[i] = f * np.exp(-1j * np.mean(
            np.angle(bg_f)
            )
            )

    return cp.asarray(f_new)



def save_video(folder, savefolder, fps = 12, quality = 10):
    """
    Saves a video to a folder. Uses maximal settings for imageio writer. fps is defined in config.
    
    savefolder = Where and name of the video.
    folder = Directory containing n_frames .png files.
    """

    #Check if package exists
    try:
        #import imageio
        import imageio.v2 as imageio
    except ImportError:
        print("Package imageio not installed. Please install with 'pip install imageio'.")
        return

    writer = imageio.get_writer(savefolder, mode = 'I', codec='mjpeg', fps=fps, quality=quality, pixelformat='yuvj444p', macro_block_size = 1)
    #writer = imageio.get_writer(savefolder, mode = 'I')

    imgs = glob.glob(folder + "*.png")
    imgs.sort(key=natural_keys)
    for file in imgs:
        im = imageio.imread(file)
        writer.append_data(im)
    writer.close()
    
def gif(folder, savefolder):
    """
    Save frames to a gif. 
    
    folder = Directory containing .png files.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Package PIL not installed. Please install with 'pip install Pillow'.")
        return

    # Create the frames
    frames = []
    imgs = glob.glob(folder + "*.png")
    imgs.sort(key=natural_keys)
    for file_name in imgs:
        new_frame = Image.open(file_name)
        new_frame = new_frame.convert("P", palette=Image.ADAPTIVE)

        frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save(savefolder, format='GIF', append_images=frames[1:] , save_all=True, duration=100, loop=0)

def conjugate_check(field):
    """
    Check if the field is conjugated. If so, conjugate it.
    """
    import numpy as np

    field = (field.get()).astype(np.complex64)
    f_new = field.copy()

    #Analyze a small region of the field in the center of the field 128x128 pixels.
    center_h = int(f_new.shape[1]/2)
    center_w = int(f_new.shape[2]/2)
    centers = np.real(f_new[:, center_h-64:center_h+64, center_w-64:center_w+64])

    #Check so previous frame and current frame are the same sign.
    for i, f in enumerate(centers[1:]):
        #Check the absolute value difference between the two frames.
        diff = np.abs(centers[i] - f)
        diff_conj = np.abs(centers[i] + f)
        if np.mean(diff) > np.mean(diff_conj):
            f_new[i] = -f_new[i]

    return cp.asarray(f_new)