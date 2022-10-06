import cv2
import scipy.ndimage as ndimage

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
