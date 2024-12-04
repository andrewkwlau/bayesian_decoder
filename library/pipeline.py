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
        smoothfactor: float = 0.2, 
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

        smoothfactor (float):
            Equivalent to smoothfactor in MATLAB smoothdata. Used to compute
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
    mouse.spikes_light, mouse.spikes_dark = u.split_lightdark(mouse.spikes_masked, mouse.darktrials)
    mouse.spikeprob_masked = u.mask_spikes(mouse.spikeprob, mouse.mask)
    mouse.spikeprob_light, mouse.spikeprob_dark = u.split_lightdark(mouse.spikeprob_masked, mouse.darktrials)


    # Get trial length
    print("2. Getting trial length.")
    mouse.trial_length = u.get_trial_length(mouse.position_mtx_masked)
    mouse.trial_light, mouse.trial_dark = u.split_lightdark(mouse.trial_length, mouse.darktrials)


    # Smooth data with Gaussian filter
    print("3. Smoothing spikes.")
    sigma = u.compute_sigma(mouse.tau, smoothfactor=smoothfactor)
    mouse.spikes_smoothed = u.gaussiansmooth_spikes(mouse.spikes_masked, mouse.mask, sigma)
    mouse.spikeprob_smoothed = u.gaussiansmooth_spikes(mouse.spikeprob_masked, mouse.mask, sigma)

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
    mouse.spikeprob_pbin = u.posbinning_data(
        mouse.spikeprob_masked,
        'spikeprob',
        mouse.position_mtx_masked,
        tunnellength,
        mouse.tau
    )
    mouse.spikeprob_pbin_smoothed = u.posbinning_data(
        mouse.spikeprob_smoothed,
        'spikeprob',
        mouse.position_mtx_masked,
        tunnellength,
        mouse.tau
    )


    # Split data into light and dark trials
    print("5. Splitting light vs dark.")
    mouse.fr_light, mouse.fr_dark = u.split_lightdark(
        mouse.fr, 
        mouse.darktrials
    )
    mouse.fr_light_smoothed, mouse.fr_dark_smoothed = u.split_lightdark(
        mouse.fr_smoothed, 
        mouse.darktrials
    )  
    mouse.spikeprob_pbin_light, mouse.spikeprob_pbin_dark = u.split_lightdark(
        mouse.spikeprob_pbin,
        mouse.darktrials
    )
    mouse.spikeprob_pbin_smoothed_light, mouse.spikeprob_pbin_smoothed_dark = u.split_lightdark(
        mouse.spikeprob_pbin_smoothed,
        mouse.darktrials
    )

    # Scale firing rates
    print("6. Scaling firing rates.")
    mouse.fr_light_scaled, mouse.fr_dark_scaled = u.scale_firingrate(
        mouse.fr_light, 
        mouse.fr_dark
    )
    mouse.fr_light_scaled_smoothed, mouse.fr_dark_scaled_smoothed = u.scale_firingrate(
        mouse.fr_light_smoothed, 
        mouse.fr_dark_smoothed
    )


def run_decoder(
        mouse: d.MouseData,
        x: int = 5, 
        tunnellength: int = 50, 
        num_pbins: int = 46, 
        smooth: bool = True,
        smoothfactor: float = 0.2, 
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

        smoothfactor (float):
            Equivalent to smoothfactor in MATLAB smoothdata. Used to compute
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
    get_tuning_curves(mouse, x, tunnellength, smoothfactor)

    # Decoder training set options
    print("7. Running decoder...")
    if smooth == True:
        training_lgtlgt = mouse.fr_light_smoothed
        training_drkdrk = mouse.fr_dark_smoothed
        if scale == True:
            training_lgtdrk = mouse.fr_light_scaled_smoothed
            training_drklgt = mouse.fr_dark_scaled_smoothed
        elif scale == False:
            training_lgtdrk = mouse.fr_light_smoothed
            training_drklgt = mouse.fr_dark_smoothed
    elif smooth == False:
        training_lgtlgt = mouse.fr_light
        training_drkdrk = mouse.fr_dark
        if scale == True:
            training_lgtdrk = mouse.fr_light_scaled
            training_drklgt = mouse.fr_dark_scaled
        elif scale == False:
            training_lgtdrk = mouse.fr_light
            training_drklgt = mouse.fr_dark
    

    # Decoder with options for smoothing and scaling of firing rates
    print("Running lgtlgt...")
    posterior_lgtlgt, decoded_pos_lgtlgt = b.bayesian_decoder(
            mouse,
            training_lgtlgt,
            mouse.spikes_light,
            num_pbins,
            uniformprior
        )
    print("lgtlgt completed.")
    print("Running drkdrk...")
    posterior_drkdrk, decoded_pos_drkdrk = b.bayesian_decoder(
            mouse,
            training_drkdrk,
            mouse.spikes_dark,
            num_pbins,
            uniformprior
        )
    print("drkdrk completed.")
    print("Running lgtdrk...")
    posterior_lgtdrk, decoded_pos_lgtdrk = b.bayesian_decoder(
        mouse,
        training_lgtdrk,
        mouse.spikes_dark,
        num_pbins,
        uniformprior
    )
    print("lgtdrk completed.")
    print("Running drklgt...")
    posterior_drklgt, decoded_pos_drklgt = b.bayesian_decoder(
        mouse,
        training_drklgt,
        mouse.spikes_light,
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
        smoothfactor: float = 0.2, 
        scale: bool = True,
        uniformprior: bool = False,
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
    get_tuning_curves(mouse, x, tunnellength, smoothfactor)

    # Sort trial start
    print("Sorting trials and chunking trials.")
    spikes_light = u.sort_trialstart(mouse.spikes_light, mouse.position_mtx, mouse.darktrials, 'light')
    spikes_dark = u.sort_trialstart(mouse.spikes_dark, mouse.position_mtx, mouse.darktrials, 'dark')
    fr_light = u.sort_trialstart(mouse.fr_light, mouse.position_mtx, mouse.darktrials, 'light')
    fr_dark = u.sort_trialstart(mouse.fr_dark, mouse.position_mtx, mouse.darktrials, 'dark')
    fr_light_smoothed = u.sort_trialstart(mouse.fr_light_smoothed, mouse.position_mtx, mouse.darktrials, 'light')
    fr_dark_smoothed = u.sort_trialstart(mouse.fr_dark_smoothed, mouse.position_mtx, mouse.darktrials, 'dark')
    fr_light_scaled = u.sort_trialstart(mouse.fr_light_scaled, mouse.position_mtx, mouse.darktrials, 'light')
    fr_dark_scaled = u.sort_trialstart(mouse.fr_dark_scaled, mouse.position_mtx, mouse.darktrials, 'dark')
    fr_light_scaled_smoothed = u.sort_trialstart(mouse.fr_light_scaled_smoothed, mouse.position_mtx, mouse.darktrials, 'light')
    fr_dark_scaled_smoothed = u.sort_trialstart(mouse.fr_dark_scaled_smoothed, mouse.position_mtx, mouse.darktrials, 'dark')
    trial_light = u.sort_trialstart(mouse.trial_light, mouse.position_mtx, mouse.darktrials, 'light')
    trial_dark = u.sort_trialstart(mouse.trial_dark, mouse.position_mtx, mouse.darktrials, 'dark')

    # Chunking trials in lists
    spikes_light_chunks = u.chunk_trials(spikes_light, num_chunks)
    spikes_dark_chunks = u.chunk_trials(spikes_dark, num_chunks)
    fr_light_chunks = u.chunk_trials(fr_light, num_chunks)
    fr_dark_chunks = u.chunk_trials(fr_dark, num_chunks)
    fr_light_smoothed_chunks = u.chunk_trials(fr_light_smoothed, num_chunks)
    fr_dark_smoothed_chunks = u.chunk_trials(fr_dark_smoothed, num_chunks)
    fr_light_scaled_chunks = u.chunk_trials(fr_light_scaled, num_chunks)
    fr_dark_scaled_chunks = u.chunk_trials(fr_dark_scaled, num_chunks)
    fr_light_scaled_smoothed_chunks = u.chunk_trials(fr_light_scaled_smoothed, num_chunks)
    fr_dark_scaled_smoothed_chunks = u.chunk_trials(fr_dark_scaled_smoothed, num_chunks)
    trial_light_chunks = u.chunk_trials(trial_light, num_chunks)
    trial_dark_chunks = u.chunk_trials(trial_dark, num_chunks)

    # Intialise output
    posterior_allchunks = []
    decoded_pos_allchunks = []


    # Decoder training set options
    print("7. Running decoder...")
    if smooth == True:
        training_lgtlgt = fr_light_smoothed_chunks
        training_drkdrk = fr_dark_smoothed_chunks
        if scale == True:
            training_lgtdrk = fr_light_scaled_smoothed_chunks
            training_drklgt = fr_dark_scaled_smoothed_chunks
        elif scale == False:
            training_lgtdrk = fr_light_smoothed_chunks
            training_drklgt = fr_dark_smoothed_chunks
    elif smooth == False:
        training_lgtlgt = fr_light_chunks
        training_drkdrk = fr_dark_chunks
        if scale == True:
            training_lgtdrk = fr_light_scaled_chunks
            training_drklgt = fr_dark_scaled_chunks
        elif scale == False:
            training_lgtdrk = fr_light_chunks
            training_drklgt = fr_dark_chunks


    # Loop through each chunk
    for i in range(num_chunks):
        print("Decoding chunk ", i, "...")
        posterior_lgtlgt, decoded_pos_lgtlgt = b.bayesian_decoder_chunks(
            mouse,
            training_lgtlgt[i],
            spikes_light_chunks[i],
            trial_light_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " lgtlgt completed.")
        posterior_drkdrk, decoded_pos_drkdrk = b.bayesian_decoder_chunks(
            mouse,
            training_drkdrk[i],
            spikes_dark_chunks[i],
            trial_dark_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " drkdrk completed.")
        posterior_lgtdrk, decoded_pos_lgtdrk = b.bayesian_decoder_chunks(
            mouse,
            training_lgtdrk[i],
            spikes_dark_chunks[i],
            trial_dark_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " lgtdrk completed.")
        posterior_drklgt, decoded_pos_drklgt = b.bayesian_decoder_chunks(
            mouse,
            training_drklgt[i],
            spikes_light_chunks[i],
            trial_light_chunks[i],
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
        smoothfactor: float = 0.2, 
        scale: bool = True,
        uniformprior: bool = False,
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
    mouse.trial_length = u.get_trial_length(mouse.position_mtx_masked)
    mouse.trial_light, mouse.trial_dark = u.split_lightdark(mouse.trial_length, mouse.darktrials)

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
        spikes_light, spikes_dark = u.split_lightdark(spikes_masked, mouse.darktrials)

        # Smooth data with Gaussian filter
        print("3. Smoothing spikes.")
        sigma = u.compute_sigma(mouse.tau, smoothfactor=smoothfactor)
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
        fr_light, fr_dark = u.split_lightdark(fr, mouse.darktrials)
        fr_light_smoothed, fr_dark_smoothed = u.split_lightdark(fr_smoothed, mouse.darktrials)

        # Scale firing rates
        print("6. Scaling firing rates.")
        fr_light_scaled, fr_dark_scaled = u.scale_firingrate(fr_light, fr_dark)
        fr_light_scaled_smoothed, fr_dark_scaled_smoothed = u.scale_firingrate(fr_light_smoothed, fr_dark_smoothed)

        # Sort trial start
        print("7. Sorting trials and chunking trials.")
        spikes_light = u.sort_trialstart(spikes_light, mouse.position_mtx, mouse.darktrials, 'light')
        spikes_dark = u.sort_trialstart(spikes_dark, mouse.position_mtx, mouse.darktrials, 'dark')
        fr_light = u.sort_trialstart(fr_light, mouse.position_mtx, mouse.darktrials, 'light')
        fr_dark = u.sort_trialstart(fr_dark, mouse.position_mtx, mouse.darktrials, 'dark')
        fr_light_smoothed = u.sort_trialstart(fr_light_smoothed, mouse.position_mtx, mouse.darktrials, 'light')
        fr_dark_smoothed = u.sort_trialstart(fr_dark_smoothed, mouse.position_mtx, mouse.darktrials, 'dark')
        fr_light_scaled = u.sort_trialstart(fr_light_scaled, mouse.position_mtx, mouse.darktrials, 'light')
        fr_dark_scaled = u.sort_trialstart(fr_dark_scaled, mouse.position_mtx, mouse.darktrials, 'dark')
        fr_light_scaled_smoothed = u.sort_trialstart(fr_light_scaled_smoothed, mouse.position_mtx, mouse.darktrials, 'light')
        fr_dark_scaled_smoothed = u.sort_trialstart(fr_dark_scaled_smoothed, mouse.position_mtx, mouse.darktrials, 'dark')
        trial_light = u.sort_trialstart(mouse.trial_light, mouse.position_mtx, mouse.darktrials, 'light')
        trial_dark = u.sort_trialstart(mouse.trial_dark, mouse.position_mtx, mouse.darktrials, 'dark')

        # Chunking trials in lists
        spikes_light_chunks = u.chunk_trials(spikes_light, num_chunks)
        spikes_dark_chunks = u.chunk_trials(spikes_dark, num_chunks)
        fr_light_chunks = u.chunk_trials(fr_light, num_chunks)
        fr_dark_chunks = u.chunk_trials(fr_dark, num_chunks)
        fr_light_smoothed_chunks = u.chunk_trials(fr_light_smoothed, num_chunks)
        fr_dark_smoothed_chunks = u.chunk_trials(fr_dark_smoothed, num_chunks)
        fr_light_scaled_chunks = u.chunk_trials(fr_light_scaled, num_chunks)
        fr_dark_scaled_chunks = u.chunk_trials(fr_dark_scaled, num_chunks)
        fr_light_scaled_smoothed_chunks = u.chunk_trials(fr_light_scaled_smoothed, num_chunks)
        fr_dark_scaled_smoothed_chunks = u.chunk_trials(fr_dark_scaled_smoothed, num_chunks)
        trial_light_chunks = u.chunk_trials(trial_light, num_chunks)
        trial_dark_chunks = u.chunk_trials(trial_dark, num_chunks)

        # Intialise output
        posterior_allchunks = []
        decoded_pos_allchunks = []

        # Decoder training set options
        print("8. Running decoder...")
        if smooth == True:
            training_lgtlgt = fr_light_smoothed_chunks
            training_drkdrk = fr_dark_smoothed_chunks
            if scale == True:
                training_lgtdrk = fr_light_scaled_smoothed_chunks
                training_drklgt = fr_dark_scaled_smoothed_chunks
            elif scale == False:
                training_lgtdrk = fr_light_smoothed_chunks
                training_drklgt = fr_dark_smoothed_chunks
        elif smooth == False:
            training_lgtlgt = fr_light_chunks
            training_drkdrk = fr_dark_chunks
            if scale == True:
                training_lgtdrk = fr_light_scaled_chunks
                training_drklgt = fr_dark_scaled_chunks
            elif scale == False:
                training_lgtdrk = fr_light_chunks
                training_drklgt = fr_dark_chunks


        # Loop through each chunk
        for i in range(num_chunks):
            print("Decoding rep ", rep, " chunk ", i, "...")
            posterior_lgtlgt, decoded_pos_lgtlgt = b.bayesian_decoder_chunks(
                mouse,
                training_lgtlgt[i],
                spikes_light_chunks[i],
                trial_light_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "lgtlgt completed.")
            posterior_drkdrk, decoded_pos_drkdrk = b.bayesian_decoder_chunks(
                mouse,
                training_drkdrk[i],
                spikes_dark_chunks[i],
                trial_dark_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "drkdrk completed.")
            posterior_lgtdrk, decoded_pos_lgtdrk = b.bayesian_decoder_chunks(
                mouse,
                training_lgtdrk[i],
                spikes_dark_chunks[i],
                trial_dark_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "lgtdrk completed.")
            posterior_drklgt, decoded_pos_drklgt = b.bayesian_decoder_chunks(
                mouse,
                training_drklgt[i],
                spikes_light_chunks[i],
                trial_light_chunks[i],
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
        mouse: d.MouseData, 
        posterior_all: dict, 
        decoded_pos_all: dict,
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
    # Confusion Matrices
    print()
    print("Confusion Matrix lgtlgt")
    confusion_mtx_lgtlgt = r.generate_confusion_mtx(
        mouse,
        decoded_pos_all['lgtlgt'],
        'lgtlgt',
        num_pbins
    )
    print()
    print("Confusion Matrix drkdrk")
    confusion_mtx_drkdrk = r.generate_confusion_mtx(
        mouse,
        decoded_pos_all['drkdrk'],
        'drkdrk',
        num_pbins
    )
    print()
    print("Confusion Matrix lgtdrk")
    confusion_mtx_lgtdrk = r.generate_confusion_mtx(
        mouse,
        decoded_pos_all['lgtdrk'],
        'lgtdrk',
        num_pbins
    )
    print()
    print("Confusion Matrix drklgt")
    confusion_mtx_drklgt = r.generate_confusion_mtx(
        mouse,
        decoded_pos_all['drklgt'],
        'drklgt',
        num_pbins
    )

    # Accuracy
    print()
    print("Accuracy lgtlgt")
    accuracy_lgtlgt = r.compute_accuracy(
        mouse,
        decoded_pos_all['lgtlgt'],
        'lgtlgt'
    )
    print()
    print("Accuracy drkdrk")
    accuracy_drkdrk = r.compute_accuracy(
        mouse,
        decoded_pos_all['drkdrk'],
        'drkdrk'
    )
    print()
    print("Accuracy lgtdrk")
    accuracy_lgtdrk = r.compute_accuracy(
        mouse,
        decoded_pos_all['lgtdrk'],
        'lgtdrk'
    )
    print()
    print("Accuracy drklgt")
    accuracy_drklgt = r.compute_accuracy(
        mouse,
        decoded_pos_all['drklgt'],
        'drklgt'
    )

    # Errors
    print()
    print("Errors lgtlgt")
    errors_lgtlgt, mse_lgtlgt, rt_mse_lgtlgt = r.compute_errors(
        mouse,
        decoded_pos_all['lgtlgt'],
        'lgtlgt'
    )
    print()
    print("Errors drkdrk")
    errors_drkdrk, mse_drkdrk, rt_mse_drkdrk = r.compute_errors(
        mouse,
        decoded_pos_all['drkdrk'],
        'drkdrk'
    )
    print()
    print("Errors lgtdrk")
    errors_lgtdrk, mse_lgtdrk, rt_mse_lgtdrk = r.compute_errors(
        mouse,
        decoded_pos_all['lgtdrk'],
        'lgtdrk'
    )
    print()
    print("Errors drklgt")
    errors_drklgt, mse_drklgt, rt_mse_drklgt = r.compute_errors(
        mouse,
        decoded_pos_all['drklgt'],
        'drklgt'
    )


    # Outputs
    confusion_mtx_all = {
        'lgtlgt': confusion_mtx_lgtlgt,
        'drkdrk': confusion_mtx_drkdrk,
        'lgtdrk': confusion_mtx_lgtdrk,
        'drklgt': confusion_mtx_drklgt
    }
    accuracy_all = {
        'lgtlgt': accuracy_lgtlgt,
        'drkdrk': accuracy_drkdrk,
        'lgtdrk': accuracy_lgtdrk,
        'drklgt': accuracy_drklgt
    }
    errors_all = {
        'lgtlgt': errors_lgtlgt,
        'drkdrk': errors_drkdrk,
        'lgtdrk': errors_lgtdrk,
        'drklgt': errors_drklgt
    }
    mse_all = {
        'lgtlgt': mse_lgtlgt,
        'drkdrk': mse_drkdrk,
        'lgtdrk': mse_lgtdrk,
        'drklgt': mse_drklgt
    }
    rt_mse_all = {
        'lgtlgt': rt_mse_lgtlgt,
        'drkdrk': rt_mse_drkdrk,
        'lgtdrk': rt_mse_lgtdrk,
        'drklgt': rt_mse_drklgt
    }

    return confusion_mtx_all, accuracy_all, errors_all, mse_all, rt_mse_all


def run_results_chunks(
        mouse: d.MouseData, 
        posterior_allchunks: dict, 
        decoded_pos_allchunks: dict,
        num_pbins: int = 46,
        num_chunks: int = 10
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
        num_chunks
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
                i
            )
            print()
            print("Errors chunk", i, paradigm, ":")
            errors, mse, rt_mse = r.compute_errors_chunk(
                mouse,
                decoded_pos_allchunks[i][paradigm],
                paradigm,
                num_chunks,
                i
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
                    i
                )
                print()
                print("Errors, Rep", rep, "chunk", i, paradigm, ":")
                errors, mse, rt_mse = r.compute_errors_chunk(
                    mouse,
                    decoded_pos_allreps[rep][i][paradigm],
                    paradigm,
                    num_chunks,
                    i
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
