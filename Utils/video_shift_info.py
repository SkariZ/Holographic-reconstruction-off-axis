# -*- coding: utf-8 -*-
"""
Created on Fri July 22 09:14:39 2021

@author: Fredrik Skärberg
"""

# Predefined video settings from when the experiment was taken.
import numpy as np

def get_video_shift_info ():
    #Check if video settings are available.

  data = np.zeros((11, 3), dtype = object)

  data[0] = ['4hoursInkubation_Every4_6', 10.454, 3]
  data[1] = ['4hoursInkubation_Every4_5', 10.3854, 3]
  data[2] = ['4hoursInkubation_Every4_4', 10.4193, 3]
  data[3] = ['4hoursInkubation_Every4_3', 10.4852, 3]
  data[4] = ['4hoursInkubation_Every4_2', 10.4947368421053, 3]
  data[5] = ['SecondSample_4hoursInkubation_new_spot_Every4_1', 12.7134, 2]
  data[6] = ['4hoursInkubation_new_spot_Every4_2', 12.7168, 3]
  data[7] = ['JAWSII_psl_1hoursIncubation_Every4_3', 12.5565, 3]
  data[8] = ['JAWSII_psl_1hoursIncubation_Every4_1', 12.4375, 3]
  data[9] = ['Hunh7_added_anotherWellEvery3_3', 16.9728, 3]
  data[10] = ['AreaTwo_redLaserEvery4_1_movie_1', 10.454, 3]
  

  #colnames = ['filename', 'vid_shift', 'frame_disp_vid']  
  #dataframe = pd.DataFrame(data, columns = colnames)
  #pd.DataFrame.to_csv(dataframe, "video_shift_info.csv")
  
  return data


def check_if_video_settings_available(data_video_info, videoname):
    """
    Checks if the videoname exists in the data settings file.
    """
    
    boolval = False
    
    for i in range(len(data_video_info)):
        if data_video_info[i, 0] in videoname:
            #if data_video_info[i, 0] in videoname.split('\\'):
            boolval = True
    
    return boolval
    
def return_available_video_settings(data_video_info, videoname):
    """
    Returns available settings for vid_shift and frame_disp_vid.
    """

    for i in range(len(data_video_info)):
        if data_video_info[i, 0] in videoname:
            #if data_video_info[i, 0] in videoname.split('\\'):
            #Return predefined values.
            vid_shift = data_video_info[i, 1]
            frame_disp_vid = data_video_info[i, 2]
                
    return vid_shift, frame_disp_vid 