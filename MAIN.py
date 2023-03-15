# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:16:41 2021

@author: Fredrik Skärberg
"""

import CONFIG
import time, numpy as np, time
import matplotlib.pyplot as plt

#Modules.
import video_to_field
import simple_plot

def init(C):
    C.check_if_file_exists()
    C.create_result_folder()
    C.print_main_settings()
    C.save_config_to_results()

#Define settings in config.py and then run MAIN.py
if __name__ == "__main__":
    START = time.time()

    C = CONFIG.main_settings()
    init(C)

    ##### Preprocess the data; Retain the optical field from the holography video.
    start_time = time.time()
    video_to_field.main()    
    print("Total elapsed time for field and phase retrieval--- %s minutes ---" % str((time.time() - start_time) / 60))

    ##### Plot frames.
    start_time = time.time()
    simple_plot.main()    
    
    print("TOTAL elapsed time for script --- %s minutes ---" % str((time.time() - START) / 60))
    
#Loading the field.
#f = np.load(f'{CONFIG.main_settings.root_folder}/{CONFIG.main_settings.project_name}/field/field.npy')

#from Utils import fft_loader
#field = fft_loader.vec_to_field_multi(
#        vecs = f, 
#        shape = (CONFIG.video_settings.height - 2*CONFIG.reconstruction_settings.cropping, CONFIG.video_settings.width - 2*CONFIG.reconstruction_settings.cropping),
#        pupil_radius=CONFIG.save_settings.pupil_radius
#    )

#from Utils import Utils_z as Z