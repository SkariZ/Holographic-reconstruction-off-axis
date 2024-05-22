# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:16:41 2021

@author: Fredrik Skärberg
"""

import CONFIG
import time, numpy as np, time
import matplotlib.pyplot as plt
import os
import cupy as cp

#Modules.
from Utils import fft_loader
from Utils import image_utils
from Utils import Utils_z
from Utils import phase_utils


def init(C):
    valid = C.check_if_file_exists()
    C.create_result_folder()
    C.print_main_settings()
    C.save_config_to_results()
    return valid

#Define settings in config.py and then run MAIN.py

#Where the movies are located.
ROOT_MOVIES_FOLDERS = ["F:/OleicAcid/dish1_compartment1/", "F:/OleicAcid/dish1_compartment2/", "F:/OleicAcid/dish1_compartment3/", "F:/OleicAcid/dish1_compartment4/"]

#Get all folders in ROOT_MOVIES_FOLDERS
FOLDERS_MOVIES = []
for ROOT_MOVIES_FOLDER in ROOT_MOVIES_FOLDERS:
    FOLDERS_MOVIE = [f for f in os.listdir(ROOT_MOVIES_FOLDER) if os.path.isdir(os.path.join(ROOT_MOVIES_FOLDER, f))]
    FOLDERS_MOVIES += FOLDERS_MOVIE

ROOT_MOVIES_FOLDERS_N = []
for ROOT_MOVIES_FOLDER in ROOT_MOVIES_FOLDERS:
    ROOT_MOVIES_FOLDERS_N += [ROOT_MOVIES_FOLDER]*len(FOLDERS_MOVIE)
ROOT_MOVIES_FOLDER = ROOT_MOVIES_FOLDERS_N


#Where to save the results.
ROOT_SAVE_FOLDER = "F:/OleicAcid/Results_new/dish1/"

if not os.path.exists(ROOT_SAVE_FOLDER):
    os.mkdir(ROOT_SAVE_FOLDER)

if __name__ == "__main__":

    #Loop over all folders.
    START = time.time()
    for k, folder in enumerate(FOLDERS_MOVIES):

        C = CONFIG.main_settings(
            filename_folder=f'{ROOT_MOVIES_FOLDER[k]}/{folder}',
            project_name=folder,
            root_folder=ROOT_SAVE_FOLDER)

        #Initialize the settings.
        valid = init(C)

        if valid:

            #Take input to skip this folder or not
            cont = input("Continue? (y/n): ")
            if cont == 'n':
                continue
            
            #Settings
            c_str = f'{ROOT_SAVE_FOLDER}/{folder}/plots'

            #Get the field.
            if CONFIG.save_settings.fft_save:
                field0 = fft_loader.vec_to_field_multi(
                        vecs = cp.load(f'{ROOT_SAVE_FOLDER}/{folder}/field/field.npy')[0:1], 
                        shape = (CONFIG.video_settings.height - 2*CONFIG.reconstruction_settings.cropping, CONFIG.video_settings.width - 2*CONFIG.reconstruction_settings.cropping),
                        pupil_radius=CONFIG.save_settings.pupil_radius
                    )[0]
            else:
                field0 = cp.load(f'{ROOT_SAVE_FOLDER}/{folder}/field/field.npy')[0]

            #Refocus field and plot.
            print("Refocusing field...")
            if CONFIG.plot_settings.plot_z:    
                l , u = CONFIG.z_propagation_settings.z_search_low, CONFIG.z_propagation_settings.z_search_high
                z = cp.linspace(l, u, CONFIG.z_propagation_settings.z_steps)
                
                if cp.abs(CONFIG.z_propagation_settings.z_prop) > 0:
                    field0 = Utils_z.refocus_field_z(field0, -CONFIG.z_propagation_settings.z_prop, padding = 256)
                
                for i, zi in enumerate(z):
                    fp = Utils_z.refocus_field_z(field0, zi, padding = 128)
                    r = fp.imag
                    r = image_utils.downsample2d(r, 2) # Downsample somwehat

                    #Save frame via annotation, else plt.imsave
                    if CONFIG.plot_settings.annotate:
                        image_utils.save_frame(r, c_str + '/prop', name = f"{i}_z_{cp.round(zi, 3)}", annotate = True, annotatename = f"z = {cp.round(zi, 3)}", dpi = CONFIG.plot_settings.DPI)
                    else:
                        image_utils.save_frame(r, c_str + '/prop', name = f"{i}_z_{cp.round(zi, 3)}")

            #Take manual input for z.
            z_manual = input("Enter z value: ")
            
            #Save the z value as a text file.
            with open(f'{ROOT_SAVE_FOLDER}/{folder}/z_manual.txt', 'w') as f:
                f.write(z_manual)

            #Load full field.
            if CONFIG.save_settings.fft_save:
                field = fft_loader.vec_to_field_multi(
                            vecs = cp.load(f'{ROOT_SAVE_FOLDER}/{folder}/field/field.npy'), 
                            shape = (CONFIG.video_settings.height - 2*CONFIG.reconstruction_settings.cropping, CONFIG.video_settings.width - 2*CONFIG.reconstruction_settings.cropping),
                            pupil_radius=CONFIG.save_settings.pupil_radius
                        )
            else:
                field = cp.load(f'{ROOT_SAVE_FOLDER}/{folder}/field/field.npy')

            #Refocus field.
            if cp.abs(float(z_manual)) > 0:
                field = cp.array([Utils_z.refocus_field_z(f, float(z_manual), padding = 128) for f in field], dtype = cp.complex64)

            image_utils.save_frame(field[0].real, c_str, name = 'optical_field_real')
            image_utils.save_frame(field[0].imag, c_str, name = 'optical_field_imag')
            image_utils.save_frame(cp.abs(field[0]), c_str, name = 'intensity_image')
            image_utils.save_frame(
                cp.angle(phase_utils.phaseunwrap_skimage(field[0], norm_phase_after=True)), 
                c_str, name = 'phase_image'
                )

            #Save refocused field.
            if CONFIG.save_settings.fft_save:
                from Utils import fft_loader
                field = fft_loader.field_to_vec_multi(fields = field, pupil_radius=CONFIG.save_settings.pupil_radius)
            
            #Save field.
            cp.save(f'{ROOT_SAVE_FOLDER}/{folder}/field/field.npy', field)

            #Delete all images in prop folder
            if CONFIG.plot_settings.plot_z:
                for filename in os.listdir(c_str + '/prop/'):
                    os.remove(c_str + '/prop/' + filename)

        #Take input to continue or not.
        cont = input("Continue? (y/n): ")
        if cont == 'n':
            break


    print("TOTAL elapsed time for script --- %s minutes ---" % str((time.time() - START) / 60))
    