import numpy as np
import scipy
import mat73
from dataclasses import dataclass


def load_matfiles(file_path):
    """
    Load files with either scipy.io.loadmat or mat73.loadmat.
    """
    try:
        # Attempt to load the .mat file using scipy
        data = scipy.io.loadmat(file_path)
        print(file_path, "loaded with scipy.io.loadmat")
        return data
    except Exception as e:
        print(f"scipy.io.loadmat failed: {e})")
        print("Trying mat73.loadmat...")
        try:
            # Attempt to load the .mat file using mat73
            data = mat73.loadmat(file_path)
            print(file_path, "loaded with mat73.loadmat")
            return data
        except Exception as e:
            print(f"mat73.loadmat also failed: {e}")
            raise


def load_data(mouse_ID):
    """
    Load necessary data files of the selected mouse:
    - time_binned_SpikeInf
    - time_binned_DiscreteSpikes
    - target_positions
    - darktrial_raw
    - del_trials
    """
    time_binned_SpikeInf = load_matfiles("../datafiles/"+ mouse_ID +"/time_binned_SpikeInf.mat")
    time_binned_DiscreteSpikes = load_matfiles("../datafiles/"+ mouse_ID +"/time_binned_DiscreteSpikes.mat")
    target_positions = load_matfiles("../datafiles/"+ mouse_ID +"/target_positions.mat")
    darktrial_raw = load_matfiles("../datafiles/"+ mouse_ID +"/darktrial_raw.mat")
    del_trials = load_matfiles("../datafiles/"+ mouse_ID +"/del_trials.mat")

    return time_binned_SpikeInf, time_binned_DiscreteSpikes, target_positions, darktrial_raw, del_trials


@dataclass
class MouseData:
    # Fields created at intialisation
    mouse_ID: str
    spikeprob: np.ndarray
    spikes: np.ndarray
    position_mtx: np.ndarray
    darktrials: np.ndarray
    deltrials: np.ndarray
    tau: float

    # Fields to be filled in as needed when running decoder pipeline
    spikeprob_posbinned: np.ndarray = None
    spikes_posbinned: np.ndarray = None

    mask: np.ndarray = None
    rewardzone: int | tuple | list = None
    position_mtx_masked: np.ndarray = None
    triallength: np.ndarray = None

    spikes_masked: np.ndarray = None
    spikes_smoothed: np.ndarray = None
    spikes_light: np.ndarray = None
    spikes_dark: np.ndarray = None

    firingrate = np.ndarray = None
    fr_light: np.ndarray = None
    fr_dark: np.ndarray = None

    fr_light_scaled: np.ndarray = None
    fr_dark_scaled: np.ndarray = None

    
    def __post_init__(self):
        self.show_data()

    def show_data(self):
        print()
        print("Data of:", self.mouse_ID)
        print()
        # Spike Probability
        print("Spike Probability:")
        print(self.spikeprob.shape)
        print("Trial x 100ms Time Bin x Neuron")
        print()
        # Discrete Spikes
        print("Discrete Spikes:")
        print(self.spikes.shape)
        print("Trial x 100ms Time Bin x Neuron")
        print()
        # Position Matrix
        print("Position Matrices:")
        print(self.position_mtx.shape)
        print("Trial x 100ms Time Bin")
        print()
        # Trial Info
        print("Dark Trials:")
        print(self.darktrials.shape)
        print("Trial, (0=light, 1=dark)")
        print()
        print("Deleted Trials:")
        print(self.deltrials.shape)
        print("Trial, (Trial_Num of deleted trials)")
        print()
        print('no. of light trials: {}'.format(np.count_nonzero(self.darktrials==0)))
        print('no. of dark trials: {}'.format(np.count_nonzero(self.darktrials)))

