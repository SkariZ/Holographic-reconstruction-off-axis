import numpy as np

#TODO add commenting etc.

def data_to_real(img):
    """
    Transforms a complex image to a real image.

    Input:
        img : complex image
    Output:
        Real image.

    """

    image = np.zeros((img.shape[0],img.shape[1],2), dtype = np.float32)
    image[..., 0] = img.real
    image[..., 1] = img.imag
    return image

def real_to_imag(img):
    return img[..., 0] + 1j*img[..., 1]


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

def field_to_vec(field, pupil_radius, mask = None):
    """

    Transforms a field to a vector given a pupil radius.
    - Can be speed up by including the mask to the function.

    """
    if not np.iscomplexobj(field):
        field = field[...,0] + 1j*field[..., 1]

    h, w = field.shape
    fft = np.fft.fftshift(np.fft.fft2(field))

    if mask is None:
        mask = create_circular_mask(h, w, radius=pupil_radius)
    
    return np.array(fft[mask], dtype = np.complex64)

def field_to_vec_multi(fields, pupil_radius, mask = None):
    """

    Transforms a field to a vector given a pupil radius.
    - Can be speed up by including the mask to the function.

    """

    if not np.iscomplexobj(fields):
        fields = fields[...,0] + 1j*fields[..., 1]

    _, h, w = fields.shape

    if mask is None:
        mask = create_circular_mask(h, w, radius=pupil_radius)    

    fvec = []
    for field in fields:
        fft = np.fft.fftshift(np.fft.fft2(field))
        vec = np.array(fft[mask])
        fvec.append(vec)
    
    return np.array(fvec, dtype = np.complex64)

def vec_to_field(vec, pupil_radius, shape, mask=None, to_real=False):
    """

    Transforms a vector to a field given pupil radius and shape.
    - Can be speed up by including the mask to the function.

    """

    if mask is None:
        mask = create_circular_mask(shape[0], shape[1], radius=pupil_radius)
    mask = np.array(mask, dtype = np.complex64)
    mask[mask == 1] = vec

    field = np.fft.ifft2(np.fft.ifftshift(mask))
    if to_real:
        field = data_to_real(field)

    return field

def vec_to_field_multi(vecs, pupil_radius, shape, mask=None, to_real = False):
    """
    Transforms a vector to a field given pupil radius and shape.
    - Can be speed up by including the mask to the function.
    """

    if mask is None:
        mask = create_circular_mask(shape[0], shape[1], radius=pupil_radius)
    mask = np.array(mask, dtype = np.complex64)

    fields = []
    for vec in vecs:
        mm = mask.copy()
        mm[mm == 1] = vec
        field = np.fft.ifft2(np.fft.ifftshift(mm))
        fields.append(field)
    fields = np.array(fields, dtype = np.complex64)

    if to_real:
        fields = np.array([data_to_real(f) for f in fields], dtype = np.float32)

    return fields


def get_length(shape, pupil_radius):
    """
    Get the length of the vector for given shape and pupil radius.
    """

    x = np.arange(shape[0]) - shape[0] / 2
    y = np.arange(shape[1]) - shape[1] / 2
    X, Y = np.meshgrid(x, y)

    X=np.reshape(np.fft.fftshift(X),(shape[0]*shape[1]))
    Y=np.reshape(np.fft.fftshift(Y),(shape[0]*shape[1]))
    
    inds=np.where(X**2+Y**2<pupil_radius**2)[0]
    return len(inds)

def exp_crop(data, pupil_radius):
    """
    Data: complex valued field.
    pupil_radius: size of pupil.
    
    """

    x = np.arange(data.shape[0]) - data.shape[0] / 2
    y = np.arange(data.shape[1]) - data.shape[1] / 2
    X, Y = np.meshgrid(x, y)

    X=np.reshape(np.fft.fftshift(X),(data.shape[0]*data.shape[1]))
    Y=np.reshape(np.fft.fftshift(Y),(data.shape[0]*data.shape[1]))
    data=np.reshape(data,(data.shape[0]*data.shape[1],))

    inds=np.where(X**2+Y**2<pupil_radius**2)[0]
    inds=np.sort(inds)
    
    outdata = [data[i] for i in inds]
    outdata=np.array(outdata)
    return outdata

def get_pupil_radius(data, shape):
    """
    Get pupil size given a vector and original data size.
    Data: vector of indeces
    shape = output shape, e.g. (64, 64)
    """
    x = np.arange(shape[0]) - shape[0] / 2
    y = np.arange(shape[1]) - shape[1] / 2
    X, Y = np.meshgrid(x, y)

    X=np.reshape(np.fft.fftshift(X),(shape[0]*shape[1]))
    Y=np.reshape(np.fft.fftshift(Y),(shape[0]*shape[1]))
    
    rho=np.sqrt(X**2+Y**2)
    rho=np.sort(rho)
    pupil_radius=rho[len(data)+1]

    return pupil_radius

def exp_expand(data, shape):
    """
    Data: vector of indeces
    shape = output shape, e.g. (64, 64)
    """

    x = np.arange(shape[0]) - shape[0] / 2
    y = np.arange(shape[1]) - shape[1] / 2
    X, Y = np.meshgrid(x, y)

    bg=np.zeros((shape[0]*shape[1],))+0j
    X=np.reshape(np.fft.fftshift(X),(shape[0]*shape[1]))
    Y=np.reshape(np.fft.fftshift(Y),(shape[0]*shape[1]))

    RHO=X**2+Y**2
    Inds=np.argsort(RHO)
    Inds=np.sort(Inds[:data.shape[0]])

    bg[Inds[:data.shape[0]]]=data
    bg=np.reshape(bg,(shape[0],shape[1]))
  
    return bg

def exp_expand_index(data, indexfile):

    cs=np.reshape(indexfile, (indexfile.shape[0]*indexfile.shape[1]))
    shape=indexfile.shape
    bg=np.zeros((shape[0]*shape[1],)) + 0j

    Inds=np.argwhere(cs==1)[:,0]
    
    bg[Inds]=data
    bg=np.reshape(bg,(shape[0], shape[1]))
  
    return bg


def exp_expand_and_crop(data, shape, pupil_radius):
    """
    
    """

    x = np.arange(shape[0]) - shape[0] / 2
    y = np.arange(shape[1]) - shape[1] / 2
    X, Y = np.meshgrid(x, y)

    bg=np.zeros((shape[0]*shape[1],)) + 0j
    X=np.reshape(np.fft.fftshift(X),(shape[0]*shape[1]))
    Y=np.reshape(np.fft.fftshift(Y),(shape[0]*shape[1]))

    RHO=X**2+Y**2
    Inds=np.argsort(RHO)
    Inds=np.sort(Inds[:data.shape[0]])
    bg[Inds[:data.shape[0]]]=data
    inds=np.where(X**2+Y**2<pupil_radius**2)[0]

    outdata = [bg[i] for i in inds]
    outdata=np.array(outdata)
    return outdata

