# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:16:41 2021

@author: Fredrik Skärberg
"""

import CONFIG as CONFIG
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
    print("Total elapsed time for plotting--- %s seconds ---" % (time.time() - start_time))


    print("TOTAL elapsed time for script --- %s minutes ---" % str((time.time() - START) / 60))
    
#Loading the field.
#f = np.load(f"Results/{DATA.project_name}/field/field.npy") #If you want to load field.
#f = np.array([simple_plot.correctfield(f[i]) for i in range(len(f))], dtype = np.complex64)


if False:
    from Utils import Utils_z

    Z = np.linspace(-5, 5, 21)
    fgz = [Utils_z.refocus_field_z(fg, zz) for zz in Z]

    for i, g in enumerate(fgz):
        plt.figure(figsize = (12,12))
        plt.imshow(g.real, cmap = 'gray')
        plt.title(Z[i])
        plt.show()

    x = [np.std(np.abs(g)) for g in fgz]

    plt.plot(x)
    plt.plot(np.argmax(x), np.max(x), 'ro')
    plt.vlines(np.argmax(x), ymax = np.max(x), ymin = np.min(x), ls = '--', color = 'r')
    print("Focus", Z[np.argmax(x)])