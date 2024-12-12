import numpy as np


def preprocess_data(loaded_data, remove_punished=False):
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
            - punished_trials
            - spikeprob_shuffled
            - spikes_shuffled

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
            - punished_trials (ndarray):
                (Trial) Indices of 
            - spikeprob_shuffled (list)
            - spikes_shuffled (list)
    """
    spikeprob, spikes, position_mtx, darktrials, deltrials, punished_trials, spikeprob_shuffled, spikes_shuffled = loaded_data

    # Align axes
    spikeprob = align_axes(spikeprob)
    spikes = align_axes(spikes)
    darktrials = align_axes(darktrials, dark=True)

    # Format position index to python
    position_mtx, deltrials = format_pos_idx(position_mtx, deltrials)

    # Flatten punished trials
    punished_trials = punished_trials.flatten()

    # Remove deleted trials
    spikeprob = remove_deltrials(spikeprob, deltrials)
    spikes = remove_deltrials(spikes, deltrials)
    position_mtx = remove_deltrials(position_mtx, deltrials)
    darktrials = remove_deltrials(darktrials, deltrials)
    punished_trials = remove_deltrials(punished_trials, deltrials)

    # Remove punished trials
    if remove_punished == True:
        spikeprob = remove_punished_trials(spikeprob, punished_trials)
        spikes = remove_punished_trials(spikes, punished_trials)
        position_mtx = remove_punished_trials(position_mtx, punished_trials)
        darktrials = remove_punished_trials(darktrials, punished_trials)

    # Remove non-NaN neurons
    spikeprob = remove_nonnan_neuron(spikeprob)
    spikes = remove_nonnan_neuron(spikes)

    # Match NaN mask
    spikes = match_nan_from_spikeprob(spikeprob, spikes)

    # Remove position matrix artefacts
    position_mtx = remove_pos_artefacts(position_mtx)

    # Pre-process shuffled data
    for rep in range(len(spikes_shuffled)):
        # Align axes
        spikeprob_shuffled[rep] = align_axes(spikeprob_shuffled[rep])
        spikes_shuffled[rep] = align_axes(spikes_shuffled[rep])
        # Remove deleted trials
        spikeprob_shuffled[rep] = remove_deltrials(spikeprob_shuffled[rep], deltrials)
        spikes_shuffled[rep] = remove_deltrials(spikes_shuffled[rep], deltrials)
        # Remove punished trials
        if remove_punished == True:
            spikeprob_shuffled[rep] = remove_punished_trials(spikeprob_shuffled[rep], punished_trials)
            spikes_shuffled[rep] = remove_punished_trials(spikes_shuffled[rep], punished_trials)
        # Remove non-NaN neurons
        spikeprob_shuffled[rep] = remove_nonnan_neuron(spikeprob_shuffled[rep])
        spikes_shuffled[rep] = remove_nonnan_neuron(spikes_shuffled[rep])
        # Match NaN mask
        # spikes_shuffled[rep] = match_nan_from_spikeprob(spikeprob_shuffled[rep], spikes_shuffled[rep])

    return spikeprob, spikes, position_mtx, darktrials, deltrials, punished_trials, spikeprob_shuffled, spikes_shuffled


def align_axes(data, dark = False):
    """
    Align data matrices axes into: Trials x ____ x Neurons.
    If data matrix is darktrials, set dark = True.
    """
    if dark == True:
        data = data.swapaxes(0,1)
    else:
        data = data.swapaxes(0,1)
        data = data.swapaxes(1,2)

    return data


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


def remove_deltrials(data, deltrials):
    """
    Remove deleted trials.
    """
    data = np.delete(data, deltrials, axis=0)
    return data


def remove_punished_trials(data, punished_trials):
    """
    Remove punished trials.
    """
    trials_to_remove = np.where(punished_trials==1)[0]
    data = np.delete(data, trials_to_remove, axis=0)
    return data


def remove_nonnan_neuron(data):
    """
    Remove non-neuron data, n=0 and n=1 (background and neuropil).

    In upstream data processing, ROI 0 and ROI 1 are not neurons. They are the 
    baseline from the background and the neuropil.
    """
    data = np.delete(data, [0,1], axis=2)
    return data


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

