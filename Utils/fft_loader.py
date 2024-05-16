import cupy as cp

#TODO add commenting etc.

def data_to_real(img):
    """
    Transforms a complex image to a real image.

    Input:
        img : complex image
    Output:
        Real image.

    """

    image = cp.zeros((img.shape[0],img.shape[1],2), dtype = cp.float32)
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

    Y, X = cp.ogrid[:h, :w]
    dist_from_center = cp.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def field_to_vec(field, pupil_radius, mask = None):
    """

    Transforms a field to a vector given a pupil radius.
    - Can be speed up by including the mask to the function.

    """
    if not cp.iscomplexobj(field):
        field = field[...,0] + 1j*field[..., 1]

    h, w = field.shape
    fft = cp.fft.fftshift(cp.fft.fft2(field))

    if mask is None:
        mask = create_circular_mask(h, w, radius=pupil_radius)
    
    return cp.array(fft[mask], dtype = cp.complex64)

def field_to_vec_multi(fields, pupil_radius, mask = None):
    """

    Transforms a field to a vector given a pupil radius.
    - Can be speed up by including the mask to the function.

    """

    if not cp.iscomplexobj(fields):
        fields = fields[...,0] + 1j*fields[..., 1]

    _, h, w = fields.shape

    if mask is None:
        mask = create_circular_mask(h, w, radius=pupil_radius)    

    fvec = []
    for field in fields:
        fft = cp.fft.fftshift(cp.fft.fft2(field))
        vec = cp.array(fft[mask])
        fvec.append(vec)
    
    return cp.array(fvec, dtype = cp.complex64)

def vec_to_field(vec, pupil_radius, shape, mask=None, to_real=False):
    """

    Transforms a vector to a field given pupil radius and shape.
    - Can be speed up by including the mask to the function.

    """

    if mask is None:
        mask = create_circular_mask(shape[0], shape[1], radius=pupil_radius)
    mask = cp.array(mask, dtype = cp.complex64)
    mask[mask == 1] = vec

    field = cp.fft.ifft2(cp.fft.ifftshift(mask))
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
    mask = cp.array(mask, dtype = cp.complex64)

    fields = []
    for vec in vecs:
        mm = mask.copy()
        mm[mm == 1] = vec
        field = cp.fft.ifft2(cp.fft.ifftshift(mm))
        fields.append(field)
    fields = cp.array(fields, dtype = cp.complex64)

    if to_real:
        fields = cp.array([data_to_real(f) for f in fields], dtype = cp.float32)

    return fields


def get_length(shape, pupil_radius):
    """
    Get the length of the vector for given shape and pupil radius.
    """

    x = cp.arange(shape[0]) - shape[0] / 2
    y = cp.arange(shape[1]) - shape[1] / 2
    X, Y = cp.meshgrid(x, y)

    X=cp.reshape(cp.fft.fftshift(X),(shape[0]*shape[1]))
    Y=cp.reshape(cp.fft.fftshift(Y),(shape[0]*shape[1]))
    
    inds=cp.where(X**2+Y**2<pupil_radius**2)[0]
    return len(inds)

def exp_crop(data, pupil_radius):
    """
    Data: complex valued field.
    pupil_radius: size of pupil.
    
    """

    x = cp.arange(data.shape[0]) - data.shape[0] / 2
    y = cp.arange(data.shape[1]) - data.shape[1] / 2
    X, Y = cp.meshgrid(x, y)

    X=cp.reshape(cp.fft.fftshift(X),(data.shape[0]*data.shape[1]))
    Y=cp.reshape(cp.fft.fftshift(Y),(data.shape[0]*data.shape[1]))
    data=cp.reshape(data,(data.shape[0]*data.shape[1],))

    inds=cp.where(X**2+Y**2<pupil_radius**2)[0]
    inds=cp.sort(inds)
    
    outdata = [data[i] for i in inds]
    outdata=cp.array(outdata)
    return outdata

def get_pupil_radius(data, shape):
    """
    Get pupil size given a vector and original data size.
    Data: vector of indeces
    shape = output shape, e.g. (64, 64)
    """
    x = cp.arange(shape[0]) - shape[0] / 2
    y = cp.arange(shape[1]) - shape[1] / 2
    X, Y = cp.meshgrid(x, y)

    X=cp.reshape(cp.fft.fftshift(X),(shape[0]*shape[1]))
    Y=cp.reshape(cp.fft.fftshift(Y),(shape[0]*shape[1]))
    
    rho=cp.sqrt(X**2+Y**2)
    rho=cp.sort(rho)
    pupil_radius=rho[len(data)+1]

    return pupil_radius

def exp_expand(data, shape):
    """
    Data: vector of indeces
    shape = output shape, e.g. (64, 64)
    """

    x = cp.arange(shape[0]) - shape[0] / 2
    y = cp.arange(shape[1]) - shape[1] / 2
    X, Y = cp.meshgrid(x, y)

    bg=cp.zeros((shape[0]*shape[1],))+0j
    X=cp.reshape(cp.fft.fftshift(X),(shape[0]*shape[1]))
    Y=cp.reshape(cp.fft.fftshift(Y),(shape[0]*shape[1]))

    RHO=X**2+Y**2
    Inds=cp.argsort(RHO)
    Inds=cp.sort(Inds[:data.shape[0]])

    bg[Inds[:data.shape[0]]]=data
    bg=cp.reshape(bg,(shape[0],shape[1]))
  
    return bg

def exp_expand_index(data, indexfile):

    cs=cp.reshape(indexfile, (indexfile.shape[0]*indexfile.shape[1]))
    shape=indexfile.shape
    bg=cp.zeros((shape[0]*shape[1],)) + 0j

    Inds=cp.argwhere(cs==1)[:,0]
    
    bg[Inds]=data
    bg=cp.reshape(bg,(shape[0], shape[1]))
  
    return bg


def exp_expand_and_crop(data, shape, pupil_radius):
    """
    
    """

    x = cp.arange(shape[0]) - shape[0] / 2
    y = cp.arange(shape[1]) - shape[1] / 2
    X, Y = cp.meshgrid(x, y)

    bg=cp.zeros((shape[0]*shape[1],)) + 0j
    X=cp.reshape(cp.fft.fftshift(X),(shape[0]*shape[1]))
    Y=cp.reshape(cp.fft.fftshift(Y),(shape[0]*shape[1]))

    RHO=X**2+Y**2
    Inds=cp.argsort(RHO)
    Inds=cp.sort(Inds[:data.shape[0]])
    bg[Inds[:data.shape[0]]]=data
    inds=cp.where(X**2+Y**2<pupil_radius**2)[0]

    outdata = [bg[i] for i in inds]
    outdata=cp.array(outdata)
    return outdata

