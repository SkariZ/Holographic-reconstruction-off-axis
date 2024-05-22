import glob
import os
import shutil
from dataclasses import dataclass, field

"""
Script to load global config. 

Change the settings in each class after own preferences.

"""

ENDSWITH = 'holo.avi'

class main_settings:
    """
    Class for storing main settings, ie. folder name and project_name.
    """
    
    def __init__(self, filename_folder, project_name, root_folder):
        """
        Initialize the class.
        """
        self.filename_folder = filename_folder
        self.project_name = project_name
        self.root_folder = root_folder    


        #The filename that ends with holography. The file we want
        try:
            self.filename_holo = [f for f in glob.glob(self.filename_folder + "/*.avi") if f.endswith(ENDSWITH)][0]
        except:
            self.filename_holo = False

        #Paths that will be constructed in rooft_folder/project_name/
        self.standard_paths = ['field', 'plots']
        self.standard_paths_plot = ['frames','prop', 'sub']

    #Have them as usual functions as of now. Class will always be initialized the init values.
    def check_if_file_exists(self):
        """
        Check if the file exists.
        """
        if not os.path.isfile(self.filename_holo):
            return False
        else:
            return True

    def create_result_folder(self):
        """
        Create the folder for the results.
        """

        #Check if root folder exists.
        if not os.path.isdir(self.root_folder):
            os.mkdir(self.root_folder)

        if not os.path.isdir(f'{self.root_folder}/{self.project_name}'):
            os.mkdir(f'{self.root_folder}/{self.project_name}')

        for path in self.standard_paths:
            if not os.path.isdir(f'{self.root_folder}/{self.project_name}/{path}'):
                os.mkdir(f'{self.root_folder}/{self.project_name}/{path}')

        #In plots_folder
        for path in self.standard_paths_plot:
            if not os.path.isdir(f'{self.root_folder}/{self.project_name}/plots/{path}'):
                os.mkdir(f'{self.root_folder}/{self.project_name}/plots/{path}')

    def print_main_settings(self):
        print(f"Project name --- {self.project_name}")

    def save_config_to_results(self):
        shutil.copy('config.py', f'{self.root_folder}/{self.project_name}/config.py')


@dataclass
class multiprocessing:

    M : bool = True

@dataclass
class video_settings:
    """
    DataClass to store video settings
    """
    
    #size height
    height : int = 1400

    #size width
    width : int = 1800
    
    #Which corner to crop in image [[],[]], upper left 1, upper right 2, lower left 3, lower right 4.
    corner : int = 2

    # If you now the period of which the camera shifts, add this here. If 0 it will be estimated. (Does only matter if you will do background subtraction later, and only for index method old and pre-, prepost)
    vid_shift : float = 5

    #Edges- Some videos have black edges, some do not. 0 = no black edges, 1 = black edges. (Do not change, keep at 1)
    edges : bool = True

    #Check for good and bad frames before choosing which indeces to take out. Some videos disrupted // gray frames for example. Recommendet to keep off.
    check_good : bool = False

@dataclass
class index_settings:
    """
    DataClass to store index settings
    """

    #Cap the maximum number of frames.
    max_frames : int = 400

    #Which frame to start processing from
    start_frame : int = 0

    #How many frames before and after the vid shift that are looked at. (Only affects index_method = old and prepost)
    frame_disp_vid : int = 10

    #Which indexes to take out beforehand. 'old', 'all' or 'pre2',...'pre5', prepost, 'every' and 'own_idx' . 'all' is 0,1,2,3..... The others are a bit special.
    index_method : str = 'all'

    #Input manually the frames you want to extract. Only works if index_method = 'own_index'
    index : list[int] = field(default_factory=lambda:[])

@dataclass
class plot_settings:
    """
    DataClass to store plot settings
    """

    #Plot all frames
    plot_all : bool = True

    #Plot z
    plot_z : bool = True

    #Plot subtraction plots
    plot_sub : bool = False

    #DPI
    DPI : int = 225

    #Downsample
    downsamplesize : int = 4

    #Annotate plots
    annotate : bool = True

    #Do movies of the plots above (.avi movie)
    movie : bool = True

    #Movie fps
    movie_fps : int = 20

    #Delete images after movie is made
    delete_images : bool = True

@dataclass
class z_propagation_settings(index_settings, video_settings):
    """
    DataClass to store z-propagation settings
    """

    #Z_prop - The z-propagation distance. If this set, then we do not estimate the focus on first frame.
    z_prop : int = 0

    #Find focus for first frame and use for all other frames later
    find_focus_first_frame : bool = False

    #Which frames to subtract for when finding focus.
    find_focus_first_frame_idx_start : int = index_settings.start_frame
    find_focus_first_frame_idx_stop : int = index_settings.start_frame#video_settings.vid_shift + index_settings.start_frame + index_settings.frame_disp_vid

    ###Z-search propagation distance.
    #low
    z_search_low : int = -10

    #high
    z_search_high : int = 10

    #Step size
    z_steps : int = 51


@dataclass
class reconstruction_settings(video_settings, index_settings):
    """
    Dataclass for reconstruction settings. Also inherite the video settings.

    """
    #Which frame to do precalculations with.
    first_frame_precalc = index_settings.start_frame

    #Do a lowpassfit of background. This is necessary in Twilight off-axis 
    lowpass_fit : bool = True

    #For lowpass filtering when doing background estimation and subraction. First one is the fourier selection filter, the other are set costumized.
    radius_lowpass : list[int] = field(default_factory=lambda: [300, 25, 25, 25]) #175, 5, 5, 5
    
    #Shift fourier peak slightly. Manually, if the fourier center is slightly off..
    correct_fourier_peak : list[int] = field(default_factory=lambda: [0, 0]) #Positive row is upward shift, positive "col" is leftward shift and vice versa

    #Additionional phase corrections. A loop.
    add_phase_corrections : int = 5

    #Do phase background fit as long as the phase background is not changing more than this value.
    correct_phase_background_tol : float = 0

    #Do phase unwrapping. Do not use.
    unwrap : bool = False

    #Fit a phase background with first frame.
    first_phase_background : bool = False

    #Cropping to remove weird edges etc.
    cropping : int = 0

    #Normalize field
    normalize_field : bool = True

    #Additional mask, either "sinc" or "jinc" , or else then nothing.
    mask_f : str = ''

    #Mask_out - Mask out regions in the fftimage that can be a source of noise.
    mask_out : bool = True

    #Sign correction. If the sign is wrong, then the field will be flipped. Force positive real part.
    sign_correction : bool = False

    #Conjugate check
    conjugate_check : bool = True

    #Do a background segmentation, and normalize such that the phase is 0 in the background.
    background_segmentation : bool = True


@dataclass
class save_settings(reconstruction_settings, video_settings):

    #Save field as vector. I.e compress the field to a vector. Saves space.
    fft_save : bool = True

    #Radius of the pupil. Only used if fft_save = True
    if reconstruction_settings.lowpass_fit:
        pupil_radius : int = 300 # OBS hardcoded for now. #reconstruction_settings.radius_lowpass[0]
    else:
        pupil_radius : int = int(max([video_settings.height - reconstruction_settings.cropping*2, video_settings.width - reconstruction_settings.cropping*2]) / 6)
