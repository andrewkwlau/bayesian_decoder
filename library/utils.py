import sys
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.abspath('../library'))
import data as d


def get_trial_start(
        mouse,
        meta_filepath
):
    """
    """
    meta = pd.read_csv(meta_filepath, sep='\t', on_bad_lines='skip')

    step = float(meta[meta['mouse_id'] == 'rand_start_int'][mouse.mouse_ID].values[0])
    start = float(meta[meta['mouse_id'] == 'rand_start'][mouse.mouse_ID].values[0])
    end = float(meta[meta['mouse_id'] == 'rand_start_lim'][mouse.mouse_ID].values[0]) + step
    
    trial_start = np.arange(start, end, step) // 10 + 1
    num_chunks = len(trial_start)

    print(f"start: {start}, end: {end}, step: {step}")
    print("trial_start locations:", trial_start)
    print("number of chunks:", num_chunks)

    return trial_start, num_chunks


def get_firstx_pos(position_mtx: np.ndarray, x: int) -> np.ndarray:
    """
    Find first x positions of each trial in the session.

    Args:
        position_mtx (np.ndarray):
            (Trial, Time Bin) position data of the animal.

        x (int):
            Number of positions to find, e.g. x=5, finds first 5 position.

    Returns:
        firstx_pos (np.ndarray):
            (Trial, x) matrix of the first x position of each trial.
    """
    num_trials, num_tbins = position_mtx.shape

    # Initialise output matrix
    firstx_pos = np.full((num_trials, x), np.nan)

    for trial in range(num_trials):
        # find the first time bin with non-nan position
        first_tbin = np.where(~np.isnan(position_mtx[trial]))[0][0]

        # find corresponding position of the first non-nan time bin
        first_pos = int(position_mtx[trial, first_tbin])

        # store first 5 positions of the trial
        firstx_pos[trial] = np.array(range(first_pos, (first_pos + x)))

    print("smallest first position:", np.min(firstx_pos[:,0]))
    print("largest first position:", np.max(firstx_pos[:,0]))
    
    return firstx_pos


def create_spikesmask(
        spikes: np.ndarray, 
        position_mtx: np.ndarray, 
        spikeprob: np.ndarray = None, 
        rewardzone: int|tuple|list = None, 
        firstx_pos: np.ndarray = None
) -> np.ndarray:
    """
    Create mask with the following options:
    - match NaNs with spike probability matrix
    - mask reward zone
    - mask first x positions of each trial and all preceding time bins.

    Args:
        spikes (np.ndarray):
            (Trial, Time Bin, Neuron) discrete spikes data.

        position_mtx (np.ndarray):
            (Trial, Time Bin) position data of the animal.

        spikeprob (np.ndarray):
            (Trial, Time Bin, Neuron) spike probability matrix.

        rewardzone (int | tuple | list):
            For int (x), mask from x (first reward zone bin) to end of trial.
            For tuples (x, y), x = first reward zone bin, y = last reward zone bin.
            For lists [a, b, c], each element in list is a reward zone position bin.

        firstx_pos (np.ndarray):
            (Trial, x) first x positions of each trial. Output of the function
            get_firstx_pos().

    Returns:
        mask (np.ndarray):
            (Trial, Time Bin, Neuron) True = data, False = masked. 
    """
    num_trials, num_tbins, num_neurons = spikes.shape

    # Initialise mask to match NaNs in spikes.
    # If spikeprob is provided, intialise mask to match NaNs in spikeprob.
    if spikeprob is None:
        mask = np.copy(~np.isnan(spikes))
    else:
        mask = np.copy(~np.isnan(spikeprob))

    # If rewardzone is not provided, skip.
    if rewardzone is None:
        pass

    # If reward zone is int, mask reward zone till the end of the trial.
    elif isinstance(rewardzone, int):
        rewardzonestart_tbins = list(zip(*np.where(position_mtx == rewardzone)))    
        for i, tbin_idx in enumerate(rewardzonestart_tbins):
            trial, tbin = tbin_idx
            mask[trial, tbin:] = False

    # If reward zone is tuple (x, y), mask reward zone between x and y.
    elif isinstance(rewardzone, tuple):
        start, end = rewardzone
        rewardzone_tbins = list(zip(*np.where((position_mtx >= start) & (position_mtx <= end))))
        for i, tbin_idx in enumerate(rewardzone_tbins):
            trial, tbin = tbin_idx
            mask[trial, tbin] = False

    # If reward zone is a list, mask every position bin in list.
    elif isinstance(rewardzone, list):
        for pos in rewardzone:
            rewardzonepos_tbins = list(zip(*np.where(position_mtx == pos)))
            for i, tbin_idx in enumerate(rewardzonepos_tbins):
                trial, tbin = tbin_idx
                mask[trial, tbin] = False

    # If firstx_pos is not provided, skip.
    if firstx_pos is None:
        pass
    # If firstx_pos is provided:
    else:
        for trial in range(num_trials):
            # Mask time bins corresponding to first x position of the trial.         
            firstx_pos_tbins = []
            for pos in firstx_pos[trial]:
                firstx_pos_tbins += list(zip(*np.where(position_mtx[trial] == pos)))
            for tbin in firstx_pos_tbins:
                mask[trial,tbin] = False           
            # Mask all preceeding time bins to the last first x position of the trial.
            last_firstx_pos_tbin = firstx_pos_tbins[-1][0]
            mask[trial, :last_firstx_pos_tbin] = False

    return mask


def mask_spikes(spikes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Set spikes value to NaNs where the mask is False.

    Args:
        spikes (np.ndarray):
            (Trial, Time Bin, Neuron) discrete spikes data.

        mask (np.ndarray):
            (Trial, Time Bin, Neuron) True = data, False = masked.

    Returns:
        spikes_masked (np.ndarray):
            (Trial, Time Bin, Neuron) discrete spikes with mask applied.
    """
    spikes_masked = np.copy(spikes)
    spikes_masked[~mask] = np.nan
    return spikes_masked


def mask_position_mtx(
        position_mtx: np.ndarray, 
        rewardzone: int|tuple|list = None, 
        firstx_pos: np.ndarray = None
) -> np.ndarray:
    """
    Mask position_mtx with the following options:
    - mask reward zone
    - mask first x positions of each trial and all preceding time bins.

    Args:
        position_mtx (np.np.ndarray):
            (Trial, Time Bin) position data of the animal.

        rewardzone (int | tuple | list):
            For int (x), mask from x (first reward zone bin) to end of trial.
            For tuples (x, y), x = first reward zone bin, y = last reward zone bin.
            For lists [a, b, c], each element in list is a reward zone position bin.

        firstx_pos (np.ndarray):
            (Trial, x) first x positions of each trial. Output of the function
            get_firstx_pos().
    
    Returns:
        position_mtx_masked (np.ndarray):
            (Trial, Time Bin) masked position data of the animal.
    """  
    num_trials, num_tbins = position_mtx.shape

    position_mtx_masked = np.copy(position_mtx)

    # If rewardzone is not provided, skip.
    if rewardzone is None:
        pass

    # If reward zone is int, mask reward zone till the end of the trial.
    elif isinstance(rewardzone, int):
        rewardzonestart_tbins = list(zip(*np.where(position_mtx == rewardzone)))    
        for i, tbin_idx in enumerate(rewardzonestart_tbins):
            trial, tbin = tbin_idx
            position_mtx_masked[trial, tbin:] = np.nan
    
    # If reward zone is tuple (x, y), mask reward zone between x and y.
    elif isinstance(rewardzone, tuple):
        start, end = rewardzone
        rewardzone_tbins = list(zip(*np.where((position_mtx >= start) & (position_mtx <= end))))
        for i, tbin_idx in enumerate(rewardzone_tbins):
            trial, tbin = tbin_idx
            position_mtx_masked[trial, tbin] = np.nan

    # If reward zone is a list, mask every position bin in list.
    elif isinstance(rewardzone, list):
        for pos in rewardzone:
            rewardzonepos_tbins = list(zip(*np.where(position_mtx == pos)))
            for i, tbin_idx in enumerate(rewardzonepos_tbins):
                trial, tbin = tbin_idx
                position_mtx_masked[trial, tbin] = np.nan
    
    # If firstx_pos is not provided, skip.
    if firstx_pos is None:
        pass
    # If firstx_pos is provided:
    else:
        for trial in range(num_trials):
            # Mask time bins corresponding to first x position of the trial.         
            firstx_pos_tbins = []
            for pos in firstx_pos[trial]:
                firstx_pos_tbins += list(zip(*np.where(position_mtx[trial] == pos)))
            for tbin in firstx_pos_tbins:
                position_mtx_masked[trial,tbin] = np.nan           
            # Mask all preceeding time bins to the last first x position of the trial.
            last_firstx_pos_tbin = firstx_pos_tbins[-1][0]
            position_mtx_masked[trial, :last_firstx_pos_tbin] = np.nan

    return position_mtx_masked


def compute_sigma(tau: float, mode: str = None, windsize: float = 1.0, SDfrac: float = 0.2) -> float:
    """
    Compute sigma (standard deviation) for Gaussian kernel smoothing.

    Args:
        tau (float):
            Size of time bin of the data in seconds, e.g. 0.1 = 100ms.

        mode (str):
            Default None. It will return the sigma by mulitplying the windsize
            in num_tbins with the smoothfactor. Other options include:
            - 'FWHM': calculate sigma using Full Width at Half Maximum constant

        windsize (float):
            Window size of the Gaussian smoothing kernel. Default as 1 second.

        SDfrac (float):
            Ratio of the standard deviation to the window width. Default 1/5, i.e. 0.2,
            equivalent to MATLAB smoothdata() function.

    Returns:
        sigma (float):
            Standard deviation of the Gaussian kernel in terms of data units. If
            data unit is 100ms, a sigma of 2 means 2 x 100ms = 200ms.
    """
    if mode == 'FWHM':
        # Window size in num_tbins divided by Full Width at Half Maximum constant 2.352
        sigma = (windsize / tau) / (2 * np.sqrt(2 * np.log(2)))
    else:
        if SDfrac > 1 or SDfrac <= 0:
            raise ValueError("smoothfactor must be positive between 0 and 1.")
        sigma = (windsize / tau) * SDfrac
    print("sigma: {}, SD/windsize ratio: {}".format(sigma, SDfrac))

    return sigma



def gaussiansmooth_spikes(spikes: np.ndarray, mask: np.ndarray, sigma: int) -> np.ndarray:
    """
    Apply Gaussian smoothing and mask the spikes data.
    
    First by masking the data trial by trial. Then apply Gaussian smoothing on
    masked trial with scipy.ndimage.gaussian_filter(). Finally, apply mask to
    the smoothed spikes matrix.

    Args:
        spikes (np.ndarray):
            (Trial, Time Bin, Neuron) discrete spikes data.

        mask (np.ndarray):
            (Trial, Time Bin, Neuron) True = data, False = masked.

        sigma (int):
            Standard deviation of Gaussian kernel in number of time bins. If 
            time bin is 100ms, sigma=2 means a 200ms kernel.

    Returns:
        spikes_smoothed (np.ndarray):
            (Trial, Time Bin, Neuron) Gaussian smoothed discrete spikes data.
    """
    num_trials, num_tbins, num_neurons = spikes.shape
    
    spikes_smoothed = np.full(spikes.shape, np.nan)

    # Neuron by Neuron, Trial by Trial:
    for neuron in range(num_neurons):
        for trial in range(num_trials):
            # apply mask
            spikes_trial_masked = spikes[trial,:,neuron][mask[trial,:,neuron]]
            # mirror/reflect along boundary and smooth
            spikes_trial_smoothed = gaussian_filter1d(spikes_trial_masked, sigma, mode='reflect')
            # store smoothed spikes
            spikes_smoothed[trial,mask[trial,:,neuron],neuron] = spikes_trial_smoothed
                
    # Apply nans to match the mask
    spikes_smoothed[~mask] = np.nan
    
    return spikes_smoothed


def shuffle_spikes(spikes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Shuffle spikes for each neuron where mask = True.

    Args:
        spikes (np.ndarray):
            (Trial, Time Bin, Neuron) discrete spikes data.

        mask (np.ndarray):
            (Trial, Time Bin, Neuron) True = data, False = masked.

    Returns:
        spikes_shuffled (np.ndarray):
            (Trial, Time Bin, Neuron) shuffled discrete spikes data.
    """
    num_trials, num_tbins, num_neurons = spikes.shape

    spikes_shuffled = np.full(spikes.shape, np.nan)

    # Neuron by Neuron:
    for neuron in range(num_neurons):
        # let original data = where mask is True
        original = spikes[:,:,neuron][mask[:,:,neuron]]
        # shuffle a copy of the original data
        shuffled = np.random.shuffle(np.copy(original))
        # store shuffled data where mask is True
        spikes_shuffled[:,:,neuron][mask[:,:,neuron]] = shuffled
        
    return spikes_shuffled



# Functions that are applicable to spike probability and discrete spikes
def posbinning_data(
        data: np.ndarray, 
        data_type: str, 
        position_mtx_masked: np.ndarray, 
        num_pbins: int, 
        tau: float
) -> np.ndarray:
    """
    Binning data by position, or in other words, generating spatial tuning curves.
    For spike probability, the output is average spike probability. For discrete 
    spikes, the output is average firing rate.

    Args:
        data (np.ndarray):
            (Trial, Time Bin, Neuron) spike probability or discrete spikes data
            to be binned by position.

        data_type (str):
            The three different data types.
            - 'spikeprob' = spike probability.
            - 'npxl' = Neuropixel data.
            - 'spikes'= discrete spikes.

        position_mtx_masked (np.ndarray):
            (Trial, Time Bin) Masked position data from the output of mask_position_mtx().

        num_pbins (int):
            Number of position bins in the full length of the tunnel.

        tau (float):
            Size of time bin in seconds, e.g. 0.1 = 100ms

    Returns:
        output (np.ndarray):
            (Trial, Position Bin, Neuron)
            - for spike probability, average spike probability binned by position.
            - for discrete spikes, average firing rate binned by position.

    """
    num_trials, num_tbins, num_neurons = data.shape

    output = np.full((num_trials, num_pbins, num_neurons), np.nan)
    
    for trial in range(num_trials):
        for pbin in range(num_pbins):
            activity = []
            # Find all the times where the animal crossed the position bin
            occurence = np.where(position_mtx_masked[trial] == pbin)[0]
            occupancy = len(occurence)

            # Append all the data from these times
            for tbin in occurence:
                activity.append(data[trial,tbin,:])
            
            # Shape either (0,) or (occupancy, neuron)
            activity = np.array(activity)

            # Specify output settings
            if data_type == 'spikeprob' or data_type == 'npxl':
                # Compute average spike probability of this position bin
                activity = np.nanmean(activity, axis=0)
                output[trial,pbin,:] = activity

            elif data_type == 'spikes':
                # Compute sum of spikes of this position bin
                sum_of_spikes = np.full((num_neurons), np.nan)
                if occupancy == 0:
                    continue
                else:
                    for neuron in range(num_neurons):
                        if np.all(np.isnan(activity[:,neuron])):
                            continue
                        else:
                            sum_of_spikes[neuron] = np.nansum(activity[:,neuron])

                # Firing rate (Hz) = spike count over time in seconds
                if occupancy == 0:
                    # Set firing rate to be NaNs for all neurons
                    firingrate = np.full((num_neurons), np.nan)
                else:
                    # Otherwise, compute firing rates element-wise for each neuron
                    firingrate = sum_of_spikes / (occupancy * tau)

                output[trial,pbin,:] = firingrate

    return output


def split_lightdark(data: np.ndarray, darktrials: np.ndarray) -> tuple:
    """
    Split data matrix into lgt and drk trials.

    Args:
        data (np.ndarray):
            (Trial, ...) spike probability or discrete spikes or position_mtx data.

        darktrials (np.ndarray):
            (Trial) Array containing trial lgt/drk info. 0 = lgt, 1 = drk.

    Returns:
        tuple: a tuple containing:
            - data_lgt (np.ndarray):
                (Trial, Time Bin, Neuron) data in lgt trials.
        
            - data_drk (np.ndarray):
                (Trial, Time Bin, Neuron) data in drk trials.
    """
    data_lgt = data[np.where(darktrials == 0)[0]]
    data_drk = data[np.where(darktrials == 1)[0]]

    return data_lgt, data_drk


def sort_trialstart(
        data: np.ndarray, 
        position_mtx: np.ndarray, 
        darktrials: np.ndarray, 
        data_condition: str
) -> np.ndarray:
    """
    Sort data matrix by trial start position, from trials with earlier start to later start.

    Args:
        data (np.ndarray):
            (Trial, Time Bin, Neuron) Spike probability or discrete spikes data.

        position_mtx (np.ndarray):
            (Trial, Time Bin) position data of the animal.

        darktrials (np.ndarray):
            (Trial) Array containing trial lgt/drk info. 0 = lgt, 1 = drk.

        data_condition (str):
            'all' or 'lgt' or 'drk'.

    Returns:
        data_sorted (np.ndarray):
            (Trial, Time Bin, Neuron) sorted version of data.
    """
    num_trials = data.shape[0]

    pos_all = position_mtx
    pos_lgt, pos_drk = split_lightdark(position_mtx, darktrials)

    start_location = []
    # Find start location of each trial and apppend them
    for trial in range(num_trials):
        if data_condition == 'all':
            trial_start = pos_all[trial, np.where(~np.isnan(pos_all[trial]))[0][0]]
        elif data_condition == 'lgt':
            trial_start = pos_lgt[trial, np.where(~np.isnan(pos_lgt[trial]))[0][0]]
        elif data_condition == 'drk':
            trial_start = pos_drk[trial, np.where(~np.isnan(pos_drk[trial]))[0][0]]        
        start_location.append(trial_start)      
    
    # Sort trial start location and generate new trial index
    new_trial_index = np.argsort(start_location)
    
    # Rearrange data with new trial index
    for trial in range(num_trials):
        data_sorted = data[new_trial_index]

    return data_sorted


def scale_firingrate(fr_lgt: np.ndarray, fr_drk: np.ndarray) -> tuple:
    """
    Scale fr_lgt with (mean fr_lgt / mean fr_drk) and fr_drk with
    (mean fr_drk / mean fr_lgt).

    Args:
        fr_lgt (np.ndarray):
            (Trial, Position Bin, Neuron) firing rates of lgt trials.

        fr_drk (np.ndarray):
            (Trial, Position Bin, Neuron) firing rates of drk trials.

    Returns:
        tuple: a tuple containing:
            - fr_lgt_scaled (np.ndarray)
            - fr_drk_scaled (np.ndarray)
    """
    num_neurons = fr_lgt.shape[2]

    # Compute scaling coefficient for each neuron
    scaling_coefficients_lgt = []
    scaling_coefficients_drk = []
    
    for i in range(num_neurons):
        mean_lgt = np.nanmean(fr_lgt[:,:,i])
        mean_drk = np.nanmean(fr_drk[:,:,i])
        
        if mean_lgt == 0 or mean_drk == 0:
            coefficient_lgt_i == 1
            coefficient_drk_i == 1
        else:
            coefficient_lgt_i = mean_drk / mean_lgt
            coefficient_drk_i = mean_lgt / mean_drk

        scaling_coefficients_lgt.append(coefficient_lgt_i)
        scaling_coefficients_drk.append(coefficient_drk_i)

    # Scale firing rates
    fr_lgt_scaled = fr_lgt * scaling_coefficients_lgt
    fr_drk_scaled = fr_drk * scaling_coefficients_drk

    # Count number of neurons with higher firing rate in drk and vice versa
    num_higher_in_drk = len([i for i in scaling_coefficients_lgt if i >= 1])            
    num_higher_in_lgt = len([i for i in scaling_coefficients_drk if i >= 1])  
    print("higher in drk: {} | {} %".format(num_higher_in_drk, (num_higher_in_drk/num_neurons)*100))
    print("higher in lgt: {} | {} %".format(num_higher_in_lgt, (num_higher_in_lgt/num_neurons)*100))
        
    return fr_lgt_scaled, fr_drk_scaled


def get_trial_length(position_mtx_masked: np.ndarray) -> np.ndarray:
    """
    Get trial length of the session.

    Args:
        position_mtx_masked (np.ndarray):
            (Trial, Time Bin, Neuron) Masked position data.

    Returns:
        trial_length (np.ndarray):
            (Trial) Trial lengths of each trial in the session.
    """
    num_trials, num_tbins = position_mtx_masked.shape
    
    trial_length = np.full((num_trials), np.nan)    
    
    for trial in range(num_trials):        
        # If trial has all nans, assume trial length to be 1 position bin.
        if np.sum(~np.isnan(position_mtx_masked[trial])) == 0:
            trial_length[trial] = 1           
        else:
            # Compute distance between first and last non-NaN position of the trial
            first_pos_idx = np.where(~np.isnan(position_mtx_masked[trial]))[0][0]
            last_pos_idx = np.where(~np.isnan(position_mtx_masked[trial]))[0][-1]            
            first_pos = position_mtx_masked[trial, first_pos_idx]
            last_pos = position_mtx_masked[trial, last_pos_idx]
            distance = last_pos - first_pos

            # trial length = num of all positions occupied, thus = distance + 1
            trial_length[trial] = distance + 1

    return trial_length   


def chunk_trials(data: np.ndarray, num_chunks: int) -> list:
    """
    Chunk data by trials using np.array_split().

    Args:
        data (np.ndarray):
            (Trial, ...) Data beginning with trial on the first axis.

        num_chunks (int):
            Number of chunks to be divided. 

    Returns:
        data_list (list):
            List of data chunks after splitting.
    """
    data_list = np.array_split(data, num_chunks, axis=0)
    
    for chunk in data_list:
        print(chunk.shape)
    
    return data_list


def sort_and_chunk(
        mouse: d.CaimData | d.NpxlData,
        data: np.ndarray,
        data_condition: str,
        discrete: bool = True,
        num_chunks: int = 10
) -> list:
    """
    Sort and chunk trials.
    Equivalent to sort_trialstart() + chunk_trials() combined, plus the extra
    option to chunk trials based on discrete start positions.
    """
    num_trials = data.shape[0]

    if isinstance(mouse, d.CaimData):
        pos_all = mouse.position_mtx
    pos_lgt = mouse.pos_lgt
    pos_drk = mouse.pos_drk

    start_location = []
    # Find start location of each trial and apppend them
    for trial in range(num_trials):
        if data_condition == 'all':
            trial_start = pos_all[trial, np.where(~np.isnan(pos_all[trial]))[0][0]]
        elif data_condition == 'lgt':
            trial_start = pos_lgt[trial, np.where(~np.isnan(pos_lgt[trial]))[0][0]]
        elif data_condition == 'drk':
            trial_start = pos_drk[trial, np.where(~np.isnan(pos_drk[trial]))[0][0]]        
        start_location.append(trial_start)      
    
    # Sort trial start location and generate new trial index
    trial_start_sorted = np.sort(start_location)
    new_trial_index = np.argsort(start_location)
    print(trial_start_sorted)
    
    # Rearrange data with new trial index
    data_sorted = data[new_trial_index]

    # Chunk trials
    # if trials have discete start location
    if discrete == True:
        data_list = []
        for start_location in np.unique(trial_start_sorted):
            # Find indices of trials that have the same start location
            indices = np.where(trial_start_sorted == start_location)[0]
            data_list.append(data_sorted[indices])
    else:
        data_list = np.array_split(data_sorted, num_chunks, axis=0)

    for chunk in data_list:
        print(chunk.shape)
    
    return data_list