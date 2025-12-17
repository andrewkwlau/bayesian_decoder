import numpy as np
from scipy.ndimage import gaussian_filter1d
import h5py
import data as d


# ------------------------------------------------------------------------------
# Trial functions
# ------------------------------------------------------------------------------

def find_trial_startloc(
        data: d.Data,
        position_mtx: np.ndarray,
) -> np.ndarray:
    """
    Find the correct start locations of each trial.
    """
    num_trials, num_tbins = position_mtx.shape
    # Initialise output
    trial_startloc = np.full((num_trials,), np.nan)
    for trial in range(num_trials):
        # Find first position bin
        first_pos = position_mtx[trial, 0]
        # Find closest correct start location given (inaccurate) first position bin
        diff =  np.abs(data.start_locations - first_pos)
        closest_start = int(data.start_locations[np.argmin(diff)])
        trial_startloc[trial] = closest_start
    return trial_startloc


def get_trial_length(
        position_mtx: np.ndarray,
) -> np.ndarray:
    """
    Find the trial length given a position matrix.
    """
    first_pos = np.nanmin(position_mtx, axis=1)
    last_pos = np.nanmax(position_mtx, axis=1)
    trial_length = last_pos - first_pos + 1
    return trial_length


def sort_and_group(
        array: np.ndarray,
        trial_startloc: np.ndarray,
) -> list:
    """
    Sort and group trials by start locations.
    """
    # Sort trials by start location
    sorted_trial_starts = np.sort(trial_startloc)
    sorted_array = array[np.argsort(trial_startloc)]
    # Intialise output
    array_grouped = []
    for start in np.unique(sorted_trial_starts):
        # Find trials with this start location
        trials = np.where(sorted_trial_starts == start)[0]
        # Group those trials
        array_grouped.append(sorted_array[trials])

    return array_grouped


# ------------------------------------------------------------------------------
# Masking functions
# ------------------------------------------------------------------------------

def get_light_section(
        position_mtx: np.ndarray,
        trial_startloc: np.ndarray,
        x: int,
) -> np.ndarray:
    """
    Get the light section in the start of the tunnel.

    Args:
        data (d.Data): Data object.
        position_mtx (np.ndarray): Positions (num_trials x num_tbins).
        x (int): Length of the light section in position bins.
    """
    num_trials, num_tbins = position_mtx.shape
    # Initialise output
    light_section =np.full((num_trials, x), np.nan)
    for trial in range(num_trials):
        start = trial_startloc[trial]
        light_section[trial] = np.arange(start, start + x)
    return light_section


def create_mask(
        position_mtx: np.ndarray,
        light_section: np.ndarray,
        rewardzone: list,
) -> np.ndarray:
    """
    Create a mask to exclude light section and reward zone.
    True: keep
    False: exclude
    """
    num_trials, num_tbins = position_mtx.shape
    # Initialise output
    mask = ~np.isnan(position_mtx)

    # Mask light section
    if light_section is not None:
        for trial in range(num_trials):
            # Find all positions before the end of the light section
            last_pos_in_light_section = light_section[trial][-1]
            all_pos_before = np.unique(position_mtx[trial][position_mtx[trial] <= last_pos_in_light_section])

            # Find tbins corresponding to those positions
            tbins = []
            for pos in all_pos_before:
                tbins += list(np.where(position_mtx[trial] == pos)[0])

            # Mask those tbins and all preceding bins
            last_tbin = tbins[-1]
            mask[trial, :last_tbin + 1] = False

    # Mask reward zone
    if rewardzone is not None:
        for pos in rewardzone:
            # Find tbins in rewardzone and mask them
            tbins = np.where(position_mtx == pos)
            mask[tbins] = False

    return mask


def apply_mask(
        array: np.ndarray,
        mask: np.ndarray,
) -> np.ndarray:
    """
    Apply mask to array.
    """
    # Masking position_mtx (2D: Trials, Tbins)
    if array.ndim == 2:
        masked_array = np.where(mask, array, np.nan)
    # Masking spikes/fr (3D: Trials, Tbins, Units)
    elif array.ndim == 3:
        mask_3d = np.repeat(mask[:, :, np.newaxis], array.shape[2], axis=2)
        masked_array = np.where(mask_3d, array, np.nan)
    return masked_array


# ------------------------------------------------------------------------------
# Smoothing functions
# ------------------------------------------------------------------------------

def compute_sigma(
        tau: float,
        window_size: float = 1.0,
        SD_ratio: float = 0.2,
) -> float:
    """
    Compute sigma for Gaussian smoothing.

    Args:
        tau (float):
            Time bin size in seconds.
        window_size (float):
            Window size of the Gaussian smoothing kernel. Default as 1 second.
        SD_ratio (float):
            Ratio of standard deviation to window size. Default 1/5, i.e. 0.2,
            equivalent to MATLAB smoothdata() function.
    """
    sigma = (window_size / tau) * SD_ratio
    return sigma


def smooth_spikes(
        spikes: np.ndarray,
        mask: np.ndarray,
        sigma: float,
) -> np.ndarray:
    """
    Smooth spikes with Gaussian filter, only on unmasked data.

    Args:
        spikes (np.ndarray):
            Spike count array (Trials x Tbins x Units).
        mask (np.ndarray):
            Boolean mask array (Trials x Tbins).
        sigma (float):
            Standard deviation for Gaussian kernel in number of time bins.
    """
    num_trials, num_tbins, num_units = spikes.shape
    # Initialise output
    smoothed_spikes = np.full((num_trials, num_tbins, num_units), np.nan)

    for trial in range(num_trials):
        # Find tbins to smooth with data mask
        tbins_to_smooth = mask[trial]
        for unit in range(num_units):
            data_to_smooth = spikes[trial, tbins_to_smooth, unit]
            # Smooth data and store in output array
            smoothed_data = gaussian_filter1d(data_to_smooth, sigma, mode='reflect')
            smoothed_spikes[trial, tbins_to_smooth, unit] = smoothed_data
    
    return smoothed_spikes


# ------------------------------------------------------------------------------
# Binning and Tuning Curves
# ------------------------------------------------------------------------------

def pos_binning(
        array: np.ndarray,
        position_mtx: np.ndarray,
        num_pbins: int,
) -> np.ndarray:
    """
    """
    num_trials, num_tbins, num_units = array.shape
    # Initialise output
    output = np.full((num_trials, num_pbins, num_units), np.nan)

    for trial in range(num_trials):
        for pbin in range(num_pbins):
            # Find tbins corresponding to this position bin
            tbins = np.where(position_mtx[trial] == pbin)[0]
            occupancy = len(tbins) # in number of tbins

            if occupancy == 0:
                continue
            else:
                activity = array[trial, tbins, :]
                output[trial, pbin, :] = np.nanmean(activity, axis=0)

    return output

# ------------------------------------------------------------------------------
# Relative position functions
# ------------------------------------------------------------------------------

def get_relative_position(
        position_mtx: np.ndarray,
        num_pbins: int
) -> np.ndarray:
    """
    """
    num_trials, num_tbins = position_mtx.shape
    bin_edges = np.linspace(0, 1, num_pbins + 1)
    # Initialise output
    relative_position = np.full(position_mtx.shape, np.nan)

    for trial in range(num_trials):
        # Compute total distance of the tunnel travelled
        first_pos = np.nanmin(position_mtx[trial])
        last_pos = np.nanmax(position_mtx[trial])
        distance = last_pos - first_pos
        if distance == 0 or np.isnan(distance):
            continue

        # Normalise the distance between each position and the first one by the total distance
        norm_dist = (position_mtx[trial] - first_pos) / distance

        # Bin the normalised distance into relative position bins
        valid = ~np.isnan(norm_dist)
        relative_position[trial, valid] = np.digitize(norm_dist[valid], bin_edges) - 1

    return relative_position


def relative_pos_binning(
        array: np.ndarray,
        rel_pos_mtx: np.ndarray,
        num_pbins: int,
) -> np.ndarray:
    """
    """
    num_trials, num_tbins, num_units = array.shape
    # Initialise output
    output = np.full((num_trials, num_pbins, num_units), np.nan)
    bin_edges = np.linspace(0, 1, num_pbins + 1)

    for trial in range(num_trials):
        for pbin in range(num_pbins):
            start = bin_edges[pbin]
            end = bin_edges[pbin + 1]
            
            # Find tbins corresponding to this position bin
            if pbin == num_pbins - 1:
                # Include the right edge for the last bin
                tbins = np.where((rel_pos_mtx[trial] >= start) & (rel_pos_mtx[trial] <= end))[0]
            else:
                tbins = np.where((rel_pos_mtx[trial] >= start) & (rel_pos_mtx[trial] < end))[0]
            occupancy = len(tbins) # in number of tbins

            if occupancy == 0:
                pass
            else:
                activity = array[trial, tbins, :]
                output[trial, pbin, :] = np.nanmean(activity, axis=0)

    return output



def save_dict_to_h5(data, filename):
    with h5py.File(filename, 'w') as f:
        for key, value in data.items():
            f.create_dataset(name=key, data=value, compression="gzip")


def load_dict_from_h5(filename):
    output = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            output[key] = f[key][:]
    return output