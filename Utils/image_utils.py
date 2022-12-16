import cv2
import scipy
import scipy.ndimage as ndimage
import numpy as np
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
    
def first_frame(video, height, width, H, W, corner=1, edges = True, index=1):
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
        image = cropping_image(image, H, W, corner)
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
    average_kernel = np.ones((kernelSize, kernelSize))

    blurred_array = scipy.signal.convolve2d(inputArray, average_kernel, mode='same')
    downsampled_array = blurred_array[::kernelSize,::kernelSize]
    return downsampled_array

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

def save_video(folder, savefolder, fps = 12, quality = 10):
    """
    Saves a video to a folder. Uses maximal settings for imageio writer. fps is defined in config.
    
    savefolder = Where and name of the video.
    folder = Directory containing n_frames .png files.
    """

    #Check if package exists
    try:
        import imageio
    except ImportError:
        print("Package imageio not installed. Please install with 'pip install imageio'.")
        return

    writer = imageio.get_writer(savefolder, mode = 'I', codec='mjpeg', fps=fps, quality=quality, pixelformat='yuvj444p', macro_block_size = 1)

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