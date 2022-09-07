# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:07:06 2021

@author: Fredrik
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:23:54 2021

@author: Fredrik Skärberg
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

#Save frame in native resolution. Can change colormap if necessary.
def save_frame(frame, folder, name, cmap = 'gray'):
    plt.imsave(f"{folder}/{name}.png", frame, cmap = 'gray')
    
def preprocess_fields(image0, image1):
    
    ono=np.ones((32,32))/(32*32)
    
    an0=np.angle(image0*np.exp(-1j))+1
    an1=np.angle(image1*np.exp(-1j))+1

    ang0=convolve2d(an0, ono,mode="same")
    ang1=convolve2d(an1, ono,mode="same")
    
    image0/=np.mean(np.abs(image0))
    image0*=np.exp(-1j*(ang0))

    image1/=np.mean(np.abs(image1))
    image1*=np.exp(-1j*(ang1))
    
    return image0, image1

def main():
   #field0 = np.load('Results/' + DATA.project_name + '/field/field.npy') 
    
    
    #f1, f2 = preprocess_fields(field0[164], field0[165])
    #reference_frame = np.zeros((f1.shape[0], f1.shape[1], 2), dtype = np.float32)
    #reference_frame[..., 0] = (f1 - f2).real
    #reference_frame[..., 1] = (f1 - f2).imag
    #np.save('ref.npy', reference_frame)
    pass
           
        
        
#Main function
if __name__ == "__main__":
    main()