import numpy as np


def preprocess_data(loaded_data):
    """
    Preprocess data with various steps:
    - align axes across data matrices
    - convert position index format from MATLAB to python
    - remove eleted trials
    - remove non-NaN 'neurons'
    - match NaNs between spikeprob and spikes
    - remove position_mtx artefacts

    Args:
        loaded_data (tuple):
            Output from data.load_data(), tuple of 5 datasets.
            - spikeprob
            - spikes
            - position_mtx
            - darktrials
            - deltrials

    Returns:
        tuple: a tuple containining:
            - spikeprob (ndarray):
                (Trial, Time Bin, Neuron) spike probability data.
            - spikes (ndarray):
                (Trial, Time Bin, Neuron) discrete spikes data.
            - position_mtx (ndarray):
                (Trial, Time Bin) position matrix.
            - darktrials (ndarray):
                (Trial) Indices of light/dark trials. Light = 0, Dark = 1.
            - deltrials (ndarray):
                (Trial) Indices of deleted trials.
    """
    spikeprob, spikes, position_mtx, darktrials, deltrials = loaded_data

    spikeprob, spikes, darktrials = align_axes(spikeprob, spikes, darktrials)
    position_mtx, deltrials = format_pos_idx(position_mtx, deltrials)
    spikeprob, spikes, position_mtx, darktrials = remove_deleted_trials(spikeprob, spikes,position_mtx, darktrials, deltrials)
    spikeprob, spikes = remove_nonnan_neuron(spikeprob, spikes)
    spikes = match_nan_from_spikeprob(spikeprob, spikes)
    position_mtx = remove_pos_artefacts(position_mtx)

    return spikeprob, spikes, position_mtx, darktrials, deltrials


def align_axes(spikeprob, spikes, darktrials):
    """
    Align data matrices axes into: Trials x ____ x Neurons.
    """
    spikeprob = spikeprob.swapaxes(0,1)
    spikeprob = spikeprob.swapaxes(1,2)

    spikes = spikes.swapaxes(0,1)
    spikes = spikes.swapaxes(1,2)

    darktrials = darktrials.swapaxes(0,1)

    return spikeprob, spikes, darktrials


def format_pos_idx(position_mtx, deltrials):
    """
    Change position bins from matlab format to python. Matlab starts from 1, python
    starts from 0. Also convert deltrials into integers to be used as indices.
    """
    # Convert position_mtx
    # print(np.nanmin(position_mtx), np.nanmax(position_mtx))
    position_mtx -= 1
    # print(np.nanmin(position_mtx), np.nanmax(position_mtx))

    # Convert deltrials
    # print(deltrials)
    deltrials = np.array([int(x - 1) for x in deltrials])
    # print(deltrials)

    return position_mtx, deltrials


def remove_deleted_trials(spikeprob, spikes, position_mtx, darktrials, deltrials):
    """
    Remove deleted trials.
    """
    spikeprob = np.delete(spikeprob, deltrials, axis=0)
    spikes = np.delete(spikes, deltrials, axis=0)
    position_mtx = np.delete(position_mtx, deltrials, axis=0)
    darktrials = np.delete(darktrials, deltrials, axis=0)

    return spikeprob, spikes, position_mtx, darktrials


def remove_nonnan_neuron(spikeprob, spikes):
    """
    Remove non-neuron data, n=0 and n=1 (background and neuropil).

    In upstream data processing, ROI 0 and ROI 1 are not neurons. They are the 
    baseline from the background and the neuropil.
    """
    spikeprob = np.delete(spikeprob, [0,1], axis=2)
    spikes = np.delete(spikes, [0,1], axis=2)

    return spikeprob, spikes


def match_nan_from_spikeprob(spikeprob, spikes):
    """
    Match NaN values between spikeprob and discspikes.

    Due to artefacts from upstream data processing pipeline, there are arbitrary
    data in discspikes when it should have been NaN. To get rid of the artefact,
    we create a NaN mask from spikeprob and apply it to discspikes.
    """
    # print("nans before matching:", np.sum(np.isnan(spikes)))

    # True = NaN
    nanmask = np.isnan(spikeprob)
    spikes[nanmask] = np.nan
        
    # print("nans after matching:", np.sum(np.isnan(spikes)))

    return spikes


def remove_pos_artefacts(position_mtx):
    """
    Remove position_mtx artefacts.

    Due to artefacts from upstream data processing pipeline, there are occasions
    in position_mtx where there are zeroes when it should have been NaNs. To get 
    rid of the artefact, we replace these zeroes with NaNs.
    """
    # List of indices with position 0
    ls_of_pos0 = list(zip(*np.where(position_mtx == 0)))

    for i in ls_of_pos0:      
        trial, pos0_tbin = i    
        # find the last non-NaN index of the trial
        last_nonnan_tbin = np.where(~np.isnan(position_mtx[trial]))[0][-1]
        
        # if the position 0 time bin is also the last non-NaN time bin of the trial,
        # which is likely an artefact, it will be replaced with NaN
        if pos0_tbin == last_nonnan_tbin:
            position_mtx[i] = np.nan

    return position_mtx

