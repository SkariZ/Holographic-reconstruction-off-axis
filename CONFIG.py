#import multiprocessing
#import numpy as np
import glob
import os
import shutil
from dataclasses import dataclass, field

"""
Script to load global config. 

Change the settings in each class after own preferences.

"""

@dataclass
class main_settings:
    """
    DataClass for storing main settings, ie. folder name and project_name.
    """
    
    #Filename shall be an .avi file with the full path.
    filename_folder : str = 'D:/CellLNPsOverNight/Cells24h_cont_Every2_1'

    #Name project where the results shall be stored.
    project_name : str = 'Cells24h_cont_Every2_1'

    #The filename that ends with holography. The file we want
    filename_holo : str = [f for f in glob.glob(filename_folder + "/*.avi") if f.endswith('holo.avi')][0] if [f for f in glob.glob(filename_folder + "/*.avi") if 
    f.endswith('holo.avi')][0] else FileNotFoundError(".avi file not found in folder // or .avi file does note end with holo.avi. Change filename...")

    #Paths that will be constructed in Results/project_name/
    standard_paths : list  = field(default_factory = lambda:['field', 'plots'])

    standard_paths_plot : list = field(default_factory = lambda:['prop', 'sub'])

    #Have them as usual functions as of now. Class will always be initialized the init values.
    def check_if_file_exists(self):
        """
        Check if the file exists.
        """
        if not os.path.isfile(self.filename_holo):
            raise FileNotFoundError(f"File {self.filename_holo} does not exist.")

    def create_result_folder(self):
        """
        Create the folder for the results.
        """
        if not os.path.isdir(f'Results/{self.project_name}'):
            os.mkdir(f'Results/{self.project_name}')

        for path in self.standard_paths:
            if not os.path.isdir(f'Results/{self.project_name}/{path}'):
                os.mkdir(f'Results/{self.project_name}/{path}')

        #In plots_folder
        for path in self.standard_paths_plot:
            if not os.path.isdir(f'Results/{self.project_name}/plots/{path}'):
                os.mkdir(f'Results/{self.project_name}/plots/{path}')


    def print_main_settings(self):
        print(f"Project name --- {self.project_name}")

    def save_config_to_results(self):
        shutil.copy('config.py', f'Results/{self.project_name}/config.py')


@dataclass
class multiprocessing:

    M : bool = True


@dataclass
class video_settings:
    """
    DataClass to store video settings
    """
    
    #size height
    height : int = 1300

    #size width
    width : int = 1700
    
    #Which corner to crop in image [[],[]], upper left 0, upper right 1, lower left 2, lower right 3.
    corner : int = 1

    # If you now the period of which the camera shifts, add this here. If 0 it will be estimated. (Does only matter if you will do background subtraction later, and only for index method old and pre-, prepost)
    vid_shift : float = 20

    #Edges- Some videos have black edges, some do not. 0 = no black edges, 1 = black edges. (Do not change, keep at 1)
    edges : bool = True

    #Check for good and bad frames before choosing which indeces to take out. Some videos disrupted // gray frames for example. Recommendet to keep off.
    check_good : bool = False


@dataclass
class index_settings:

    #Cap the maximum number of frames.
    max_frames : int = 180

    #Which frame to start processing from
    start_frame : int = 0

    #How many frames before and after the vid shift that are looked at. (Only affects index_method = old and prepost)
    frame_disp_vid : int = 4 

    #Which indexes to take out beforehand. 'old', 'all' or 'pre2',...'pre5', prepost, 'every' and 'own_idx' . 'all' is 0,1,2,3..... The others are a bit special.
    index_method : str = 'all'

    #Input manually the frames you want to extract. Only works if index_method = 'own_index'
    index : list[int] = field(default_factory=lambda: [ ])

@dataclass
class plot_settings:

    #Plot all frames
    plot_all : bool = True

    #Plot z
    plot_z : bool = True

    #Plot subtraction plots
    plot_sub : bool = True

    #DPI
    DPI : int = 200

    #Downsample
    downsamplesize : int = 3

@dataclass
class z_propagation_settings(index_settings, video_settings):

    #Z_prop - The z-propagation distance. If this set, then we do not estimate the focus on first frame.
    z_prop : float = -3

    #Find focus for first frame and use for all other frames later
    find_focus_first_frame : bool = False

    #Which frames to subtract for when finding focus.
    find_focus_first_frame_idx_start : int = index_settings.start_frame
    find_focus_first_frame_idx_stop : int = video_settings.vid_shift + index_settings.start_frame + index_settings.frame_disp_vid

    ###Z-search propagation distance.
    #low
    z_search_low : int = -10
    #high
    z_search_high : int = 10


@dataclass
class reconstruction_settings(video_settings, index_settings):
    """
    Dataclass for reconstruction settings. Also inherite the video settings.

    """
    #Which frame to do precalculations with.
    first_frame_precalc = index_settings.start_frame

    #Do a lowpassfit of background. This is necessary in Twilight 
    lowpass_fit : bool = True

    #For lowpass filtering when doing background estimation and subraction. First one is the fourier selection filter, the other are set costumized.
    radius_lowpass : list[int] = field(default_factory=lambda: [200, 5, 5, 5]) #175, 5, 5, 5
    
    #Shift fourier peak slightly. Manually, if the fourier center is slightly off..
    correct_fourier_peak : list[int] = field(default_factory=lambda: [0, 0]) #Positive row is upward shift, positive "col" is leftward shift and vice versa

    #Additionional phase corrections. A loop.
    add_phase_corrections : int = 3

    #Do phase unwrapping
    unwrap : bool = False

    #Fit a phase background with first frame.
    first_phase_background : bool = False

    #Cropping to remove weird edges etc.
    cropping : int = 50
