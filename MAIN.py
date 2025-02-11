# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:16:41 2021

@author: Fredrik Skärberg
"""

import CONFIG
import time, numpy as np, time
import matplotlib.pyplot as plt
import os

#Modules.
import video_to_field
import simple_plot

def init(C):
    valid = C.check_if_file_exists()
    if not valid:
        print(f"File {C.filename_holo} does not exist.")
        return False
    C.create_result_folder()
    C.print_main_settings()
    C.save_config_to_results()
    return valid

#Define settings in config.py and then run MAIN.py

#Where the movies are located.
ROOT_MOVIES_FOLDERS = [
    "F:/YeastFlowWithBSA/Flowing_wBSAEvery1_1", 
    "F:/YeastFlowWithBSA/Flowing_wBSAEvery1_2", 
    "F:/YeastFlowWithBSA/Flowing_wBSAEvery1_3", 
    "F:/YeastFlowWithBSA/Flowing_wBSAEvery1_4", 
    "F:/YeastFlowWithBSA/Flowing_wBSAEvery1_5"
    ]

ROOT_MOVIES_FOLDERS = os.listdir('D:/Hepatocytes')

#Join with the root folder.
ROOT_MOVIES_FOLDERS = [f'D:/Hepatocytes/{f}' for f in ROOT_MOVIES_FOLDERS]

#Get all folders in ROOT_MOVIES_FOLDERS
FOLDERS_MOVIES = []
ROOT_MOVIES_FOLDERS_N = []
for ROOT_MOVIES_FOLDER in ROOT_MOVIES_FOLDERS:
    FOLDERS_MOVIE = [f for f in os.listdir(ROOT_MOVIES_FOLDER)]
    FOLDERS_MOVIES += FOLDERS_MOVIE

    curr_len = len(FOLDERS_MOVIE)
    ROOT_MOVIES_FOLDERS_N += [ROOT_MOVIES_FOLDER]*curr_len

ROOT_MOVIES_FOLDERS = ROOT_MOVIES_FOLDERS_N

#Where to save the results.
ROOT_SAVE_FOLDER = "D:/Hepatocytes/Results/"

if not os.path.exists(ROOT_SAVE_FOLDER):
    os.mkdir(ROOT_SAVE_FOLDER)

if __name__ == "__main__":

    #Loop over all folders.
    START = time.time()
    for k, folder in enumerate(FOLDERS_MOVIES):
        
        #If folder already exists, skip.
        if os.path.exists(f'{ROOT_SAVE_FOLDER}/{folder}'):
            continue

        C = CONFIG.main_settings(
            filename_folder=f'{ROOT_MOVIES_FOLDERS[k]}/{folder}',
            project_name=folder,
            root_folder=ROOT_SAVE_FOLDER)
        print(f'{ROOT_MOVIES_FOLDERS[k]}/{folder}')
        #Initialize the settings.
        valid = init(C)

        if valid:
            ##### Preprocess the data; Retain the optical field from the holography video.
            start_time = time.time()
            video_to_field.main(
                filename_holo=C.filename_holo,
                project_name=C.project_name,
                root_folder=C.root_folder,
            )    
            print("Total elapsed time for field and phase retrieval--- %s minutes ---" % str((time.time() - start_time) / 60))

            ##### Plot frames.
            start_time = time.time()
            simple_plot.main(
                filename_holo=C.filename_holo,
                project_name=C.project_name,
                root_folder=C.root_folder,
                )    
        
    print("TOTAL elapsed time for script --- %s minutes ---" % str((time.time() - START) / 60))
    
#Loading the field.
#f = np.load(f'{C.root_folder}/{C.project_name}/field/field.npy')

#from Utils import fft_loader
#field = fft_loader.vec_to_field_multi(
#        vecs = f, 
#        shape = (CONFIG.video_settings.height - 2*CONFIG.reconstruction_settings.cropping, CONFIG.video_settings.width - 2*CONFIG.reconstruction_settings.cropping),
#        pupil_radius=CONFIG.save_settings.pupil_radius
#    )

#from Utils import Utils_z as Z

#Reload module
#import importlib
#importlib.reload(u)
#from Utils import image_utils as u

#seg = u.background_segmentation(field)
#norm = u.background_normalize(field, seg)