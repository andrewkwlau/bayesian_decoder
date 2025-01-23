import sys
import os
import numpy as np

sys.path.append(os.path.abspath('../library'))
import data as d
import utils as u
import bayes as b
import results as r


def get_tuning_curves(
        mouse: d.MouseData,
        x: int = 5, 
        tunnellength: int = 50, 
        SDsize: float = 0.2, 
):
    """
    Wrapper function for getting tuning curves. Does the following for both 
    spikes and spikeprob:
    - Masking data and position matrix
    - Get trial length inforamtion for the data
    - Smooth data
    - Position binning data and generate tuning curves / firing rates matrix.
    - Split data into light and dark trials.
    - Scale data by a coefficient (only spikes/firing rates, not for spikeprob).

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        x (int):
            The first x position of the tunnel for masking spikes and position data.

        tunnellength (int):
            Number of position bins in full tunnel (including reward zone) for
            position binnning the spikes into spatial tuning curves.

        SDsize (float):
            Equivalent to SDsize in MATLAB smoothdata. Used to compute
            gaussian kernel standard deviation (sigma).

    Returns:
        None. All outputs are stored into the MouseData class.

    """
    # Masking spikes and position matrix
    print("1. Masking spikes and position matrix.")
    mouse.firstx_pos = u.get_firstx_pos(mouse.position_mtx, x)
    mouse.mask = u.create_spikesmask(
        mouse.spikes, 
        mouse.position_mtx, 
        mouse.spikeprob, 
        mouse.rewardzone, 
        mouse.firstx_pos
    )    
    mouse.position_mtx_masked = u.mask_position_mtx(
        mouse.position_mtx, 
        mouse.rewardzone, 
        mouse.firstx_pos
    )  
    mouse.spikes_masked = u.mask_spikes(mouse.spikes, mouse.mask)
    
    # Get trial length
    print("2. Getting trial length.")
    mouse.triallength = u.get_trial_length(mouse.position_mtx_masked)
    mouse.triallength_lgt, mouse.triallength_drk = u.split_lightdark(mouse.triallength, mouse.darktrials)

    # Smooth data with Gaussian filter
    print("3. Smoothing spikes.")
    sigma = u.compute_sigma(mouse.tau, SDsize=SDsize)
    mouse.spikes_smoothed = u.gaussiansmooth_spikes(mouse.spikes_masked, mouse.mask, sigma)

    # Position binning and generating firing rates
    print("4. Position Binning data and generating firing rates.")
    mouse.fr = u.posbinning_data(
        mouse.spikes_masked, 
        'spikes', 
        mouse.position_mtx_masked, 
        tunnellength, 
        mouse.tau
    )
    mouse.fr_smoothed = u.posbinning_data(
        mouse.spikes_smoothed, 
        'spikes', 
        mouse.position_mtx_masked, 
        tunnellength, 
        mouse.tau
    )

    # Split data into light and dark trials
    print("5. Splitting light vs dark.")
    mouse.pos_lgt, mouse.pos_drk = u.split_lightdark(
        mouse.position_mtx,
        mouse.darktrials
    )
    mouse.pos_lgt_masked, mouse.pos_drk_masked = u.split_lightdark(
        mouse.position_mtx_masked, 
        mouse.darktrials
    )
    mouse.spikes_lgt, mouse.spikes_drk = u.split_lightdark(
        mouse.spikes_masked, 
        mouse.darktrials
    )
    mouse.fr_lgt, mouse.fr_drk = u.split_lightdark(
        mouse.fr, 
        mouse.darktrials
    )
    mouse.fr_lgt_smoothed, mouse.fr_drk_smoothed = u.split_lightdark(
        mouse.fr_smoothed, 
        mouse.darktrials
    )  

    # Scale firing rates
    print("6. Scaling firing rates.")
    mouse.fr_lgt_scaled, mouse.fr_drk_scaled = u.scale_firingrate(
        mouse.fr_lgt, 
        mouse.fr_drk
    )
    mouse.fr_lgt_scaled_smoothed, mouse.fr_drk_scaled_smoothed = u.scale_firingrate(
        mouse.fr_lgt_smoothed, 
        mouse.fr_drk_smoothed
    )


def get_tuning_curves_npxl(
        mouse: d.NpxlData,
        x: int = 5,
        tunnellength: int = 50,
):
    """
    """
    # Find first x position
    mouse.firstx_pos_lgt = u.get_firstx_pos(mouse.pos_lgt, x)
    mouse.firstx_pos_drk = u.get_firstx_pos(mouse.pos_drk, x)

    # Create mask
    mouse.mask_lgt = u.create_spikesmask(
        mouse.spikes_lgt, 
        mouse.pos_lgt, 
        None, 
        mouse.rewardzone, 
        mouse.firstx_pos_lgt
    )
    mouse.mask_drk = u.create_spikesmask(
        mouse.spikes_drk, 
        mouse.pos_drk, 
        None, 
        mouse.rewardzone, 
        mouse.firstx_pos_drk
    )
    
    # Mask position_mtx
    mouse.pos_lgt_masked = u.mask_position_mtx(
        mouse.pos_lgt, 
        mouse.rewardzone, 
        mouse.firstx_pos_lgt
    )
    mouse.pos_drk_masked = u.mask_position_mtx(
        mouse.pos_drk, 
        mouse.rewardzone, 
        mouse.firstx_pos_drk
    )

    # Mask spikes and convolved spikes (firing rates)
    mouse.spikes_lgt_masked = u.mask_spikes(mouse.spikes_lgt, mouse.mask_lgt)
    mouse.spikes_drk_masked = u.mask_spikes(mouse.spikes_drk, mouse.mask_drk)
    mouse.fr_lgt_masked = u.mask_spikes(mouse.fr_lgt, mouse.mask_lgt)
    mouse.fr_drk_masked = u.mask_spikes(mouse.fr_drk, mouse.mask_drk)

    # Get trial lengths
    mouse.triallength_lgt = u.get_trial_length(mouse.pos_lgt_masked)
    mouse.triallength_drk = u.get_trial_length(mouse.pos_drk_masked)

    # Bin firing rates data by position
    mouse.fr_lgt_smoothed = u.posbinning_data(
        mouse.fr_lgt_masked, 
        'npxl',
        mouse.pos_lgt_masked,
        tunnellength,
        mouse.tau
    )
    mouse.fr_drk_smoothed = u.posbinning_data(
        mouse.fr_drk_masked, 
        'npxl',
        mouse.pos_drk_masked,
        tunnellength,
        mouse.tau
    )

    # Scale firing rates
    mouse.fr_lgt_scaled_smoothed, mouse.fr_drk_scaled_smoothed = u.scale_firingrate(
        mouse.fr_lgt_smoothed, 
        mouse.fr_drk_smoothed
    )


def sort_and_chunk(
        mouse: d.MouseData | d.NpxlData,
        data: np.ndarray, 
        data_condition: str,
        discrete: bool = True,
        num_chunks: int = 10
):
    """
    Sort and chunk trials.
    Equivalent to sort_trialstart() + chunk_trials() combined, plus the extra
    option to chunk trials based on discrete start positions.
    """
    num_trials = data.shape[0]

    if type(mouse) == d.MouseData:
        pos_all = mouse.position_mtx
    pos_lgt = mouse.pos_lgt
    pos_drk = mouse.pos_drk

    start_location = []
    # Find start location of each trial and apppend them
    for trial in range(num_trials):
        if data_condition == 'all':
            trial_start = pos_all[trial, np.where(~np.isnan(pos_all[trial]))[0][0]]
        elif data_condition == 'light':
            trial_start = pos_lgt[trial, np.where(~np.isnan(pos_lgt[trial]))[0][0]]
        elif data_condition == 'dark':
            trial_start = pos_drk[trial, np.where(~np.isnan(pos_drk[trial]))[0][0]]        
        start_location.append(trial_start)      
    
    # Sort trial start location and generate new trial index
    trial_start_sorted = np.sort(start_location)
    new_trial_index = np.argsort(start_location)
    print(trial_start_sorted)
    
    # Rearrange data with new trial index
    for trial in range(num_trials):
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



def run_decoder(
        mouse: d.MouseData | d.NpxlData,
        x: int = 5, 
        tunnellength: int = 50, 
        num_pbins: int = 46, 
        smooth: bool = True,
        SDsize: float = 0.2, 
        scale: bool = True,
        uniformprior: bool = False
) -> tuple:
    """
    Run normal decoder pipeline.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        x (int):
            The first x position of the tunnel for masking spikes and position data.

        tunnellength (int):
            Number of position bins in full tunnel (including reward zone) for
            position binnning the spikes into spatial tuning curves.

        num_pbins (int):
            Number of position bins to decode (excluding the reward zone).

        smooth (bool):
            Whether spikes are smoothed to generate firing rates. Default True.

        SDsize (float):
            Equivalent to SDsize in MATLAB smoothdata. Used to compute
            gaussian kernel standard deviation (sigma).

        scale (bool):
            Whether firing rates are scaled between light and dark when decoding
            with cross-training paradigms. Default True.

        uniformprior (bool):
            Whether the decoder takes a uniform prior of 1/num_pbins. Default is
            False, and the prior will vary by trial length.

    Returns:
        tuple: a tuple containing:
            - posterior_all (dict):
                a dict containing the posterior (np.ndarray): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'
            - decoded_pos_all (dict):
                a dict containing the decoded positions (np.ndarray):
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

    """
    # Run wrapper for getting tunning curves
    if type(mouse) == d.MouseData:
        get_tuning_curves(mouse, x, tunnellength, SDsize)
    elif type(mouse) == d.NpxlData:
        get_tuning_curves_npxl(mouse, x, tunnellength)

    # Decoder training set options
    print("7. Running decoder...")
    if smooth == True:
        training_lgtlgt = mouse.fr_lgt_smoothed
        training_drkdrk = mouse.fr_drk_smoothed
        if scale == True:
            training_lgtdrk = mouse.fr_lgt_scaled_smoothed
            training_drklgt = mouse.fr_drk_scaled_smoothed
        elif scale == False:
            training_lgtdrk = mouse.fr_lgt_smoothed
            training_drklgt = mouse.fr_drk_smoothed
    elif smooth == False:
        training_lgtlgt = mouse.fr_lgt
        training_drkdrk = mouse.fr_drk
        if scale == True:
            training_lgtdrk = mouse.fr_lgt_scaled
            training_drklgt = mouse.fr_drk_scaled
        elif scale == False:
            training_lgtdrk = mouse.fr_lgt
            training_drklgt = mouse.fr_drk
    

    # Decoder with options for smoothing and scaling of firing rates
    print("Running lgtlgt...")
    posterior_lgtlgt, decoded_pos_lgtlgt = b.bayesian_decoder(
            mouse,
            training_lgtlgt,
            mouse.spikes_lgt,
            num_pbins,
            uniformprior
        )
    print("lgtlgt completed.")
    print("Running drkdrk...")
    posterior_drkdrk, decoded_pos_drkdrk = b.bayesian_decoder(
            mouse,
            training_drkdrk,
            mouse.spikes_drk,
            num_pbins,
            uniformprior
        )
    print("drkdrk completed.")
    print("Running lgtdrk...")
    posterior_lgtdrk, decoded_pos_lgtdrk = b.bayesian_decoder(
        mouse,
        training_lgtdrk,
        mouse.spikes_drk,
        num_pbins,
        uniformprior
    )
    print("lgtdrk completed.")
    print("Running drklgt...")
    posterior_drklgt, decoded_pos_drklgt = b.bayesian_decoder(
        mouse,
        training_drklgt,
        mouse.spikes_lgt,
        num_pbins,
        uniformprior
    )
    print("drklgt completed.")

    # Output
    posterior_all = {
        'lgtlgt': posterior_lgtlgt,
        'drkdrk': posterior_drkdrk,
        'lgtdrk': posterior_lgtdrk,
        'drklgt': posterior_drklgt
    }   
    decoded_pos_all = {
        'lgtlgt': decoded_pos_lgtlgt,
        'drkdrk': decoded_pos_drkdrk,
        'lgtdrk': decoded_pos_lgtdrk,
        'drklgt': decoded_pos_drklgt
    }
    return posterior_all, decoded_pos_all


def run_decoder_chunks(
        mouse: d.MouseData, 
        x: int = 5,
        tunnellength: int = 50, 
        num_pbins: int = 46, 
        smooth: bool = True,
        SDsize: float = 0.2, 
        scale: bool = True,
        uniformprior: bool = False,
        discrete: bool = True,
        num_chunks: int = 10
) -> tuple:
    """
    Run chunking decoder pipeline.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        x (int):
            The first x position of the tunnel for masking spikes and position data.

        tunnellength (int):
            Number of position bins in full tunnel (including reward zone) for
            position binnning the spikes into spatial tuning curves.

        num_pbins (int):
            Number of position bins to decode (excluding the reward zone).

        smooth (bool):
            Whether spikes are smoothed to generate firing rates. Default True.

        scale (bool):
            Whether firing rates are scaled between light and dark when decoding
            with cross-training paradigms. Default True.

        uniformprior (bool):
            Whether the decoder takes a uniform prior of 1/num_pbins. Default is
            False, and the prior will vary by trial length.

        num_chunks (int):
            Number of chunks to be divided.

    Returns:
        tuple: a tuple containing:
            - posterior_allchunks (list):
                a list of all chunks' output, each chunk a dict containing
                the posterior (np.ndarray): 'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'
            - decoded_pos_allchunks (list):
                a list of all chunks' output, each chunk a dict containing
                the decoded positions (np.ndarray): 'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

    """
    # Run wrapper for getting tunning curves
    get_tuning_curves(mouse, x, tunnellength, SDsize)

    # Sort and chunk trials in list
    print("Sorting trials and chunking trials.")
    spikes_lgt_chunks = u.sort_and_chunk(mouse.spikes_lgt, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
    spikes_drk_chunks = u.sort_and_chunk(mouse.spikes_drk, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
    fr_lgt_chunks = u.sort_and_chunk(mouse.fr_lgt, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
    fr_drk_chunks = u.sort_and_chunk(mouse.fr_drk, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
    fr_lgt_smoothed_chunks = u.sort_and_chunk(mouse.fr_lgt_smoothed, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
    fr_drk_smoothed_chunks = u.sort_and_chunk(mouse.fr_drk_smoothed, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
    fr_lgt_scaled_chunks = u.sort_and_chunk(mouse.fr_lgt_scaled, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
    fr_drk_scaled_chunks = u.sort_and_chunk(mouse.fr_drk_scaled, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
    fr_lgt_scaled_smoothed_chunks = u.sort_and_chunk(mouse.fr_lgt_scaled_smoothed, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
    fr_drk_scaled_smoothed_chunks = u.sort_and_chunk(mouse.fr_drk_scaled_smoothed, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
    triallength_lgt_chunks = u.sort_and_chunk(mouse.triallength_lgt, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
    triallength_drk_chunks = u.sort_and_chunk(mouse.triallength_drk, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)


    # Intialise output
    posterior_allchunks = []
    decoded_pos_allchunks = []

    # Decoder training set options
    print("7. Running decoder...")
    if smooth == True:
        training_lgtlgt = fr_lgt_smoothed_chunks
        training_drkdrk = fr_drk_smoothed_chunks
        if scale == True:
            training_lgtdrk = fr_lgt_scaled_smoothed_chunks
            training_drklgt = fr_drk_scaled_smoothed_chunks
        elif scale == False:
            training_lgtdrk = fr_lgt_smoothed_chunks
            training_drklgt = fr_drk_smoothed_chunks
    elif smooth == False:
        training_lgtlgt = fr_lgt_chunks
        training_drkdrk = fr_drk_chunks
        if scale == True:
            training_lgtdrk = fr_lgt_scaled_chunks
            training_drklgt = fr_drk_scaled_chunks
        elif scale == False:
            training_lgtdrk = fr_lgt_chunks
            training_drklgt = fr_drk_chunks


    # Loop through each chunk
    for i in range(num_chunks):
        print("Decoding chunk ", i, "...")
        posterior_lgtlgt, decoded_pos_lgtlgt = b.bayesian_decoder_chunks(
            mouse,
            training_lgtlgt[i],
            spikes_lgt_chunks[i],
            triallength_lgt_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " lgtlgt completed.")
        posterior_drkdrk, decoded_pos_drkdrk = b.bayesian_decoder_chunks(
            mouse,
            training_drkdrk[i],
            spikes_drk_chunks[i],
            triallength_drk_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " drkdrk completed.")
        posterior_lgtdrk, decoded_pos_lgtdrk = b.bayesian_decoder_chunks(
            mouse,
            training_lgtdrk[i],
            spikes_drk_chunks[i],
            triallength_drk_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " lgtdrk completed.")
        posterior_drklgt, decoded_pos_drklgt = b.bayesian_decoder_chunks(
            mouse,
            training_drklgt[i],
            spikes_lgt_chunks[i],
            triallength_lgt_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " drklgt completed.")

        # Output
        posterior_chunk = {
            'lgtlgt': posterior_lgtlgt,
            'drkdrk': posterior_drkdrk,
            'lgtdrk': posterior_lgtdrk,
            'drklgt': posterior_drklgt
        }     
        decoded_pos_chunk = {
            'lgtlgt': decoded_pos_lgtlgt,
            'drkdrk': decoded_pos_drkdrk,
            'lgtdrk': decoded_pos_lgtdrk,
            'drklgt': decoded_pos_drklgt
        }
        posterior_allchunks.append(posterior_chunk)
        decoded_pos_allchunks.append(decoded_pos_chunk)

    return posterior_allchunks, decoded_pos_allchunks


def run_decoder_chance(
        mouse: d.MouseData, 
        num_reps: int,
        x: int = 5,
        tunnellength: int = 50, 
        num_pbins: int = 46, 
        smooth: bool = True,
        SDsize: float = 0.2, 
        scale: bool = True,
        uniformprior: bool = False,
        discrete: bool = True,
        num_chunks: int = 10
) -> tuple:
    """
    Run decoder in chunks for chance level estimation using shuffled data.
    """
    # Intialise output
    posterior_allreps = []
    decoded_pos_allreps = []

    # Masking position matrix and getting trial lengths
    print("1. Masking position matrix and getting trial lengths.")
    mouse.firstx_pos = u.get_firstx_pos(mouse.position_mtx, x)
    mouse.position_mtx_masked = u.mask_position_mtx(
        mouse.position_mtx, 
        mouse.rewardzone, 
        mouse.firstx_pos
    )
    mouse.triallength = u.get_trial_length(mouse.position_mtx_masked)
    mouse.triallength_lgt, mouse.triallength_drk = u.split_lightdark(mouse.triallength, mouse.darktrials)

    for rep in range(num_reps):
        print("Rep ", rep, " start...")

        spikes = mouse.spikes_shuffled[rep]
        spikeprob = mouse.spikeprob_shuffled[rep]

        # Masking spikes and position matrix
        print("2. Masking spikes.")    
        mask = u.create_spikesmask(
            spikes, 
            mouse.position_mtx, 
            spikeprob, 
            mouse.rewardzone, 
            mouse.firstx_pos
        )
        spikes_masked = u.mask_spikes(spikes, mask)
        spikes_lgt, spikes_drk = u.split_lightdark(spikes_masked, mouse.darktrials)

        # Smooth data with Gaussian filter
        print("3. Smoothing spikes.")
        sigma = u.compute_sigma(mouse.tau, SDsize=SDsize)
        spikes_smoothed = u.gaussiansmooth_spikes(spikes_masked, mask, sigma)

        # Position binning and generating firing rates
        print("4. Position Binning data and generating firing rates.")
        fr = u.posbinning_data(
            spikes_masked, 
            'spikes', 
            mouse.position_mtx_masked, 
            tunnellength, 
            mouse.tau
        )
        fr_smoothed = u.posbinning_data(
            spikes_smoothed, 
            'spikes', 
            mouse.position_mtx_masked, 
            tunnellength, 
            mouse.tau
        )

        # Split data into light and dark trials
        print("5. Splitting light vs dark.")
        fr_lgt, fr_drk = u.split_lightdark(fr, mouse.darktrials)
        fr_lgt_smoothed, fr_drk_smoothed = u.split_lightdark(fr_smoothed, mouse.darktrials)

        # Scale firing rates
        print("6. Scaling firing rates.")
        fr_lgt_scaled, fr_drk_scaled = u.scale_firingrate(fr_lgt, fr_drk)
        fr_lgt_scaled_smoothed, fr_drk_scaled_smoothed = u.scale_firingrate(fr_lgt_smoothed, fr_drk_smoothed)
       
        # Sort and chunk trials in list
        print("7. Sorting trials and chunking trials.")
        spikes_lgt_chunks = u.sort_and_chunk(spikes_lgt, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
        spikes_drk_chunks = u.sort_and_chunk(spikes_drk, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
        fr_lgt_chunks = u.sort_and_chunk(fr_lgt, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
        fr_drk_chunks = u.sort_and_chunk(fr_drk, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
        fr_lgt_smoothed_chunks = u.sort_and_chunk(fr_lgt_smoothed, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
        fr_drk_smoothed_chunks = u.sort_and_chunk(fr_drk_smoothed, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
        fr_lgt_scaled_chunks = u.sort_and_chunk(fr_lgt_scaled, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
        fr_drk_scaled_chunks = u.sort_and_chunk(fr_drk_scaled, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
        fr_lgt_scaled_smoothed_chunks = u.sort_and_chunk(fr_lgt_scaled_smoothed, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
        fr_drk_scaled_smoothed_chunks = u.sort_and_chunk(fr_drk_scaled_smoothed, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)
        triallength_lgt_chunks = u.sort_and_chunk(mouse.triallength_lgt, mouse.position_mtx, mouse.darktrials, 'light', 10, discrete=discrete)
        triallength_drk_chunks = u.sort_and_chunk(mouse.triallength_drk, mouse.position_mtx, mouse.darktrials, 'dark', 10, discrete=discrete)


        # Intialise output
        posterior_allchunks = []
        decoded_pos_allchunks = []

        # Decoder training set options
        print("8. Running decoder...")
        if smooth == True:
            training_lgtlgt = fr_lgt_smoothed_chunks
            training_drkdrk = fr_drk_smoothed_chunks
            if scale == True:
                training_lgtdrk = fr_lgt_scaled_smoothed_chunks
                training_drklgt = fr_drk_scaled_smoothed_chunks
            elif scale == False:
                training_lgtdrk = fr_lgt_smoothed_chunks
                training_drklgt = fr_drk_smoothed_chunks
        elif smooth == False:
            training_lgtlgt = fr_lgt_chunks
            training_drkdrk = fr_drk_chunks
            if scale == True:
                training_lgtdrk = fr_lgt_scaled_chunks
                training_drklgt = fr_drk_scaled_chunks
            elif scale == False:
                training_lgtdrk = fr_lgt_chunks
                training_drklgt = fr_drk_chunks


        # Loop through each chunk
        for i in range(num_chunks):
            print("Decoding rep ", rep, " chunk ", i, "...")
            posterior_lgtlgt, decoded_pos_lgtlgt = b.bayesian_decoder_chunks(
                mouse,
                training_lgtlgt[i],
                spikes_lgt_chunks[i],
                triallength_lgt_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "lgtlgt completed.")
            posterior_drkdrk, decoded_pos_drkdrk = b.bayesian_decoder_chunks(
                mouse,
                training_drkdrk[i],
                spikes_drk_chunks[i],
                triallength_drk_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "drkdrk completed.")
            posterior_lgtdrk, decoded_pos_lgtdrk = b.bayesian_decoder_chunks(
                mouse,
                training_lgtdrk[i],
                spikes_drk_chunks[i],
                triallength_drk_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "lgtdrk completed.")
            posterior_drklgt, decoded_pos_drklgt = b.bayesian_decoder_chunks(
                mouse,
                training_drklgt[i],
                spikes_lgt_chunks[i],
                triallength_lgt_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "drklgt completed.")

            # Output
            posterior_chunk = {
                'lgtlgt': posterior_lgtlgt,
                'drkdrk': posterior_drkdrk,
                'lgtdrk': posterior_lgtdrk,
                'drklgt': posterior_drklgt
            }     
            decoded_pos_chunk = {
                'lgtlgt': decoded_pos_lgtlgt,
                'drkdrk': decoded_pos_drkdrk,
                'lgtdrk': decoded_pos_lgtdrk,
                'drklgt': decoded_pos_drklgt
            }
            posterior_allchunks.append(posterior_chunk)
            decoded_pos_allchunks.append(decoded_pos_chunk)
        
        posterior_allreps.append(posterior_allchunks)
        decoded_pos_allreps.append(decoded_pos_allchunks)
        print("Completed Rep ", rep)

    print("Completed all reps.")
    
    return posterior_allreps, decoded_pos_allreps


def run_results(
        mouse: d.MouseData | d.NpxlData, 
        num_pbins: int = 46
) -> tuple:
    """
    Run confusion matrices, accuracy, and errors.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.
        
        posterior_all (dict):
            A dictionary of posterior outputs for all training paradigms:
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

        decoded_pos_all (dict):
            A dictionary of decoded positions outputs for all trianing paradigms:
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

        num_pbins (int):
            Number of position bins to generate confusion matrix (excluding the reward zone).

    Returns:
        tuple: a tuple containing:
            - confusion_mtx_all (dict):
                a dict containing the confusion matrices (np.ndarray): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

            - accuracy_all (dict):
                a dict containing the accuracy values (float): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

            - errors_all (dict):
                a dict containing the errors (np.ndarray): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

            - mse_all (dict):
                a dict containing the Mean Squared Errors (float): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

            - rt_mse_all (dict):
                a dict containing the Root Mean Squared Errors (float): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'
    """  
    
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']
    confusion_mtx = {paradigm:[] for paradigm in paradigms}
    accuracy = {paradigm:[] for paradigm in paradigms}
    errors = {paradigm:[] for paradigm in paradigms}
    mse = {paradigm:[] for paradigm in paradigms}
    rt_mse = {paradigm:[] for paradigm in paradigms}
    
    
    # Confusion Matrices
    print()
    print("Confusion Matrix lgtlgt")
    confusion_mtx['lgtlgt'] = r.generate_confusion_mtx(
        mouse,
        mouse.decoded_pos_all['lgtlgt'],
        'lgtlgt',
        num_pbins
    )
    print()
    print("Confusion Matrix drkdrk")
    confusion_mtx['drkdrk'] = r.generate_confusion_mtx(
        mouse,
        mouse.decoded_pos_all['drkdrk'],
        'drkdrk',
        num_pbins
    )
    print()
    print("Confusion Matrix lgtdrk")
    confusion_mtx['lgtdrk'] = r.generate_confusion_mtx(
        mouse,
        mouse.decoded_pos_all['lgtdrk'],
        'lgtdrk',
        num_pbins
    )
    print()
    print("Confusion Matrix drklgt")
    confusion_mtx['drklgt'] = r.generate_confusion_mtx(
        mouse,
        mouse.decoded_pos_all['drklgt'],
        'drklgt',
        num_pbins
    )

    # Accuracy
    print()
    print("Accuracy lgtlgt")
    accuracy['lgtlgt'] = r.compute_accuracy(
        mouse,
        mouse.decoded_pos_all['lgtlgt'],
        'lgtlgt'
    )
    print()
    print("Accuracy drkdrk")
    accuracy['drkdrk'] = r.compute_accuracy(
        mouse,
        mouse.decoded_pos_all['drkdrk'],
        'drkdrk'
    )
    print()
    print("Accuracy lgtdrk")
    accuracy['lgtdrk'] = r.compute_accuracy(
        mouse,
        mouse.decoded_pos_all['lgtdrk'],
        'lgtdrk'
    )
    print()
    print("Accuracy drklgt")
    accuracy['drklgt'] = r.compute_accuracy(
        mouse,
        mouse.decoded_pos_all['drklgt'],
        'drklgt'
    )

    # Errors
    print()
    print("Errors lgtlgt")
    errors['lgtlgt'], mse['lgtlgt'], rt_mse['lgtlgt'] = r.compute_errors(
        mouse,
        mouse.decoded_pos_all['lgtlgt'],
        'lgtlgt'
    )
    print()
    print("Errors drkdrk")
    errors['drkdrk'], mse['drkdrk'], rt_mse['drkdrk'] = r.compute_errors(
        mouse,
        mouse.decoded_pos_all['drkdrk'],
        'drkdrk'
    )
    print()
    print("Errors lgtdrk")
    errors['lgtdrk'], mse['lgtdrk'], rt_mse['lgtdrk'] = r.compute_errors(
        mouse,
        mouse.decoded_pos_all['lgtdrk'],
        'lgtdrk'
    )
    print()
    print("Errors drklgt")
    errors['drklgt'], mse['drklgt'], rt_mse['drklgt'] = r.compute_errors(
        mouse,
        mouse.decoded_pos_all['drklgt'],
        'drklgt'
    )

    # Outputs
    results = {
        'confusion_mtx': confusion_mtx,
        'accuracy': accuracy,
        'errors': errors,
        'mse': mse,
        'rt_mse': rt_mse
    }
    return results


def run_results_chunks(
        mouse: d.MouseData, 
        posterior_allchunks: dict, 
        decoded_pos_allchunks: dict,
        num_pbins: int = 46,
        num_chunks: int = 10,
        discrete: bool = True
) -> tuple:
    """
    Run confusion matrices, accuracy, and errors.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.
        
        posterior_all (dict):
            A dictionary of posterior outputs for all training paradigms:
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

        decoded_pos_all (dict):
            A dictionary of decoded positions outputs for all trianing paradigms:
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

        num_pbins (int):
            Number of position bins to generate confusion matrix (excluding the reward zone).

    Returns:
        dict: a dict containing:
            - confusion_mtx_allchunks (dict):
                a dict containing the confusion matrices (np.ndarray): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

            - mean_accuracy (dict):
                a dict containing the mean accuracy (float) across chunks of each paradigm

            - mean_error (dict):
                a dict containing the mean error (flaot) across chunks of each paradigm

            - accuracy_allchunks (list):
                a list containing the accuracy values (float) in each chunk (dict): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

            - errors_allchunks (list):
                a list containing the errors (np.ndarray) in each chunk (dict): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

            - mse_allchunks (list):
                a list containing the Mean Squared Errors (float) in each chunk (dict): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

            - rt_mse_allchunks (list):
                a list containing the Root Mean Squared Errors (float) in each chunk (dict): 
                'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'
    """  
    # Confusion Matrices
    confusion_mtx_allchunks = r.generate_confusion_mtx_allchunks(
        mouse,
        decoded_pos_allchunks,
        num_pbins,
        num_chunks,
        discrete=discrete
    )

    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']

    # Initialise dict for storing accuracy and error values for all chunks
    accuracy_allchunks = {paradigm:[] for paradigm in paradigms}
    errors_allchunks = {paradigm:[] for paradigm in paradigms}
    mse_allchunks = {paradigm:[] for paradigm in paradigms}
    rt_mse_allchunks = {paradigm:[] for paradigm in paradigms}

    # Compute accuracy and errors for each chunk
    for i in range(num_chunks):
        for paradigm in paradigms:
            print("Accuracy chunk", i, paradigm, ":")
            accuracy = r.compute_accuracy_chunk(
                mouse,
                decoded_pos_allchunks[i][paradigm],
                paradigm,
                num_chunks,
                i,
                discrete=discrete
            )
            print()
            print("Errors chunk", i, paradigm, ":")
            errors, mse, rt_mse = r.compute_errors_chunk(
                mouse,
                decoded_pos_allchunks[i][paradigm],
                paradigm,
                num_chunks,
                i,
                discrete=discrete
            )
            accuracy_allchunks[paradigm].append(accuracy)
            errors_allchunks[paradigm] += errors.tolist()
            mse_allchunks[paradigm].append(mse)
            rt_mse_allchunks[paradigm].append(rt_mse)

    # Compute and store mean accuracy and mean error across chunks
    mean_accuracy = {paradigm:[] for paradigm in paradigms}
    mean_error = {paradigm:[] for paradigm in paradigms}
    for paradigm in paradigms:
        mean_accuracy[paradigm].append(np.nanmean(accuracy_allchunks[paradigm]))
        mean_accuracy[paradigm] = mean_accuracy[paradigm][0]
        mean_error[paradigm].append(np.nanmean(errors_allchunks[paradigm]))
        mean_error[paradigm] = mean_error[paradigm][0]
        

    results = {
        'confusion_mtx': confusion_mtx_allchunks,
        'mean_accuracy': mean_accuracy,
        'mean_error': mean_error,
        'accuracy_allchunks': accuracy_allchunks,
        'errors_allchunks': errors_allchunks,
        'mse_allchunks': mse_allchunks,
        'rt_mse_allchunks': rt_mse_allchunks
    }  
    return results


def run_results_chance(
        mouse: d.MouseData, 
        posterior_allreps: dict, 
        decoded_pos_allreps: dict,
        num_reps: int,
        num_pbins: int = 46,
        num_chunks: int = 10,
        discrete: bool = True
):
    """
    """
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']

    # Intialise dict for storing mean accuracy and mean error for each rep
    accuracy_allreps = {paradigm:[] for paradigm in paradigms}
    errors_allreps = {paradigm:[] for paradigm in paradigms}

    for rep in range(num_reps):
        # Initialise dict for storing accuracy and error values for all chunks
        accuracy_allchunks = {paradigm:[] for paradigm in paradigms}
        errors_allchunks = {paradigm:[] for paradigm in paradigms}
        mse_allchunks = {paradigm:[] for paradigm in paradigms}
        rt_mse_allchunks = {paradigm:[] for paradigm in paradigms}

        # Compute accuracy and errors for each chunk
        for i in range(num_chunks):
            for paradigm in paradigms:
                print("Accuracy, Rep", rep, "chunk", i, paradigm, ":")
                accuracy = r.compute_accuracy_chunk(
                    mouse,
                    decoded_pos_allreps[rep][i][paradigm],
                    paradigm,
                    num_chunks,
                    i,
                    discrete=discrete
                )
                print()
                print("Errors, Rep", rep, "chunk", i, paradigm, ":")
                errors, mse, rt_mse = r.compute_errors_chunk(
                    mouse,
                    decoded_pos_allreps[rep][i][paradigm],
                    paradigm,
                    num_chunks,
                    i,
                    discrete=discrete
                )
                accuracy_allchunks[paradigm].append(accuracy)
                errors_allchunks[paradigm] += errors.tolist()
                mse_allchunks[paradigm].append(mse)
                rt_mse_allchunks[paradigm].append(rt_mse)

        # Compute and store mean accuracy and mean error across chunks for each rep
        for paradigm in paradigms:
            accuracy_allreps[paradigm].append(np.nanmean(accuracy_allchunks[paradigm]))
            errors_allreps[paradigm].append(np.nanmean(errors_allchunks[paradigm]))

    # Compute mean accuracy and mean error across reps
    mean_accuracy_allreps = {paradigm:[] for paradigm in paradigms}
    mean_error_allreps = {paradigm:[] for paradigm in paradigms}
    for paradigm in paradigms:
        mean_accuracy_allreps[paradigm] = np.nanmean(accuracy_allreps[paradigm])
        mean_error_allreps[paradigm] = np.nanmean(errors_allreps[paradigm])

    results = {
        'mean_accuracy': mean_accuracy_allreps,
        'mean_error': mean_error_allreps,
        'accuracy_allreps': accuracy_allreps,
        'errors_allreps': errors_allreps,
    } 
    return results
