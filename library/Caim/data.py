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


def load_data(mouse_ID, tau=None):
    """
    Load necessary data files of the selected mouse:
    - time_binned_SpikeInf
    - time_binned_DiscreteSpikes
    - target_positions
    - darktrial_raw
    - del_trials
    - punished_trials
    - Shuffled_time_binned_SpikeInf
    - Shuffled_time_binned_DiscreteSpikes

    Args:
        mouse_ID
        tau (float):
            Default None. If tau=0.2, load data for 200ms time bins.
    """
    if tau == None or tau == 0.1:
        time_binned_SpikeInf = load_matfiles("../datafiles/"+ mouse_ID +"/time_binned_SpikeInf.mat")
        time_binned_DiscreteSpikes = load_matfiles("../datafiles/"+ mouse_ID +"/time_binned_DiscreteSpikes.mat")
        target_positions = load_matfiles("../datafiles/"+ mouse_ID +"/target_positions.mat")
        Shuffled_time_binned_SpikeInf = load_matfiles("../datafiles/"+ mouse_ID +"/Shuffled_time_binned_SpikeInf_200msbins.mat")
        Shuffled_time_binned_DiscreteSpikes = load_matfiles("../datafiles/"+ mouse_ID +"/Shuffled_time_binned_DiscreteSpikes_200msbins.mat")
    elif tau >= 0.2:
        ms = str(int(tau * 1000))
        time_binned_SpikeInf = load_matfiles("../datafiles/"+ mouse_ID +"/time_binned_SpikeInf_"+ ms +"msbins.mat")
        time_binned_DiscreteSpikes = load_matfiles("../datafiles/"+ mouse_ID +"/time_binned_DiscreteSpikes_"+ ms +"msbins.mat")
        target_positions = load_matfiles("../datafiles/"+ mouse_ID +"/target_positions_"+ ms +"msbins.mat")
        Shuffled_time_binned_SpikeInf = load_matfiles("../datafiles/"+ mouse_ID +"/Shuffled_time_binned_SpikeInf_"+ ms +"msbins.mat")
        Shuffled_time_binned_DiscreteSpikes = load_matfiles("../datafiles/"+ mouse_ID +"/Shuffled_time_binned_DiscreteSpikes_"+ ms +"msbins.mat")
               
    darktrial_raw = load_matfiles("../datafiles/"+ mouse_ID +"/darktrial_raw.mat")
    del_trials = load_matfiles("../datafiles/"+ mouse_ID +"/del_trials.mat")
    punished_trials = load_matfiles("../datafiles/"+ mouse_ID +"/punished_trials.mat")
    
    # Extracting data matrix from dictionary
    spikeprob = time_binned_SpikeInf['time_Caimg']
    spikes = time_binned_DiscreteSpikes['time_Caimg']
    position_mtx = target_positions['position_tbins']
    deltrials = del_trials['del_trials']
    punished_trials = punished_trials['punished_trial']
    spikeprob_shuffled = Shuffled_time_binned_SpikeInf['Shuffled_time_Caimg']
    spikes_shuffled = Shuffled_time_binned_DiscreteSpikes['Shuffled_time_Caimg']

    # Handling inconsistency in MATLAB data keys and matrix shape for darktrials
    if mouse_ID == 'C57_60_Octavius':
        darktrials = darktrial_raw['darktrial_raw'].reshape(1, darktrial_raw['darktrial_raw'].shape[0])
    elif mouse_ID == 'C57_61_Priamus':
        darktrials = np.array(darktrial_raw['darktrial']).reshape(1, len(darktrial_raw['darktrial']))
    else:
        darktrials = darktrial_raw['darktrial']

    return spikeprob, spikes, position_mtx, darktrials, deltrials, punished_trials, spikeprob_shuffled, spikes_shuffled


@dataclass
class CaimData:
    # Fields created at intialisation
    mouse_ID: str
    tau: float
    rewardzone: int | tuple | list
    spikeprob: np.ndarray
    spikes: np.ndarray
    position_mtx: np.ndarray
    darktrials: np.ndarray
    deltrials: np.ndarray
    punished_trials: np.ndarray
    spikeprob_shuffled: list
    spikes_shuffled: list

    def __post_init__(self):
        self.show_data()

    def show_data(self):
        print()
        print("Data of:", self.mouse_ID)
        print()
        # Spike Probability
        print("Spike Probability:")
        print(self.spikeprob.shape)
        print("Trial x Time Bin x Neuron")
        print()
        # Discrete Spikes
        print("Discrete Spikes:")
        print(self.spikes.shape)
        print("Trial x Time Bin x Neuron")
        print()
        # Position Matrix
        print("Position Matrices:")
        print(self.position_mtx.shape)
        print("Trial x Time Bin")
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


@dataclass
class NpxlData:
    mouse_ID: str
    tau: float
    rewardzone: int | tuple | list