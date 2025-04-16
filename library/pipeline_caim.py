import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

sys.path.append(os.path.abspath('../library'))
import data as d
import utils as u
import bayes as b
import results as r


def get_tuning_curves(
        mouse: d.CaimData,
        x: int = 5, 
        tunnellength: int = 50, 
        SDfrac: float = 0.2, 
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

        SDfrac (float):
            Equivalent to SDfrac in MATLAB smoothdata. Used to compute
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
    mouse.spikeprob_masked = u.mask_spikes(mouse.spikeprob, mouse.mask)
    
    # Get trial length
    print("2. Getting trial length.")
    mouse.triallength = u.get_trial_length(mouse.position_mtx_masked)

    # Smooth data with Gaussian filter
    print("3. Smoothing spikes.")
    sigma = u.compute_sigma(mouse.tau, SDfrac=SDfrac)
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
    mouse.pos_lgt, mouse.pos_drk = u.split_lightdark(
        mouse.position_mtx,
        mouse.darktrials
    )
    mouse.pos_lgt_masked, mouse.pos_drk_masked = u.split_lightdark(
        mouse.position_mtx_masked, 
        mouse.darktrials
    )
    mouse.triallength_lgt, mouse.triallength_drk = u.split_lightdark(
        mouse.triallength, mouse.darktrials
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
    mouse.spikeprob_fr_lgt, mouse.spikeprob_fr_drk = u.split_lightdark(
        mouse.spikeprob_pbin,
        mouse.darktrials
    )
    mouse.spikeprob_fr_lgt_smoothed, mouse.spikeprob_fr_drk_smoothed = u.split_lightdark(
        mouse.spikeprob_pbin_smoothed,
        mouse.darktrials
    )

    # Scale firing rates
    print("6. Scaling firing rates.")
    mouse.fr_lgt_scaled_smoothed, mouse.fr_drk_scaled_smoothed = u.scale_firingrate(
        mouse.fr_lgt_smoothed, 
        mouse.fr_drk_smoothed
    )


def run_decoder(
        mouse: d.CaimData,
        x: int = 5, 
        tunnellength: int = 50, 
        num_pbins: int = 46, 
        SDfrac: float = 0.2, 
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

        SDfrac (float):
            Equivalent to SDfrac in MATLAB smoothdata. Used to compute
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
    if type(mouse) == d.CaimData | d.NpxlData:
        get_tuning_curves(mouse, x, tunnellength, SDfrac)
    elif type(mouse) == d.NpxlData:
        get_tuning_curves_npxl(mouse, x, tunnellength)

    # Decoder training set options
    print("7. Running decoder...")
    training_lgtlgt = mouse.fr_lgt_smoothed
    training_drkdrk = mouse.fr_drk_smoothed
    if scale == True:
        training_lgtdrk = mouse.fr_lgt_scaled_smoothed
        training_drklgt = mouse.fr_drk_scaled_smoothed
    elif scale == False:
        training_lgtdrk = mouse.fr_lgt_smoothed
        training_drklgt = mouse.fr_drk_smoothed

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
        mouse: d.CaimData | d.NpxlData, 
        x: int = 5,
        tunnellength: int = 50, 
        num_pbins: int = 46, 
        SDfrac: float = 0.2, 
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
    get_tuning_curves(mouse, x, tunnellength, SDfrac)

    # Sort and chunk trials in list
    print("Sorting trials and chunking trials.")
    spikes_lgt_chunks = u.sort_and_chunk(mouse, mouse.spikes_lgt, 'lgt', discrete, num_chunks)
    spikes_drk_chunks = u.sort_and_chunk(mouse, mouse.spikes_drk, 'drk', discrete, num_chunks)
    fr_lgt_smoothed_chunks = u.sort_and_chunk(mouse, mouse.fr_lgt_smoothed, 'lgt', discrete, num_chunks)
    fr_drk_smoothed_chunks = u.sort_and_chunk(mouse, mouse.fr_drk_smoothed, 'drk', discrete, num_chunks)
    fr_lgt_scaled_smoothed_chunks = u.sort_and_chunk(mouse, mouse.fr_lgt_scaled_smoothed, 'lgt', discrete, num_chunks)
    fr_drk_scaled_smoothed_chunks = u.sort_and_chunk(mouse, mouse.fr_drk_scaled_smoothed, 'drk', discrete, num_chunks)
    triallength_lgt_chunks = u.sort_and_chunk(mouse, mouse.triallength_lgt, 'lgt', discrete, num_chunks)
    triallength_drk_chunks = u.sort_and_chunk(mouse, mouse.triallength_drk, 'drk', discrete, num_chunks)

    # Intialise output
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']
    posterior = {paradigm : [] for paradigm in paradigms}
    decoded_pos = {paradigm : [] for paradigm in paradigms}

    # Decoder training set options
    print("7. Running decoder...")
    training_lgtlgt = fr_lgt_smoothed_chunks
    training_drkdrk = fr_drk_smoothed_chunks
    if scale == True:
        training_lgtdrk = fr_lgt_scaled_smoothed_chunks
        training_drklgt = fr_drk_scaled_smoothed_chunks
    elif scale == False:
        training_lgtdrk = fr_lgt_smoothed_chunks
        training_drklgt = fr_drk_smoothed_chunks

    # Loop through each chunk
    for i in range(num_chunks):
        posterior_chunk = {paradigm: [] for paradigm in paradigms}
        decoded_pos_chunk = {paradigm: [] for paradigm in paradigms}

        print("Decoding chunk ", i, "...")
        posterior_chunk['lgtlgt'], decoded_pos_chunk['lgtlgt'] = b.bayesian_decoder_chunks(
            mouse,
            training_lgtlgt[i],
            spikes_lgt_chunks[i],
            triallength_lgt_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " lgtlgt completed.")
        posterior_chunk['drkdrk'], decoded_pos_chunk['drkdrk'] = b.bayesian_decoder_chunks(
            mouse,
            training_drkdrk[i],
            spikes_drk_chunks[i],
            triallength_drk_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " drkdrk completed.")
        posterior_chunk['lgtdrk'], decoded_pos_chunk['lgtdrk'] = b.bayesian_decoder_chunks(
            mouse,
            training_lgtdrk[i],
            spikes_drk_chunks[i],
            triallength_drk_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " lgtdrk completed.")
        posterior_chunk['drklgt'], decoded_pos_chunk['drklgt'] = b.bayesian_decoder_chunks(
            mouse,
            training_drklgt[i],
            spikes_lgt_chunks[i],
            triallength_lgt_chunks[i],
            num_pbins,
            uniformprior
        )
        print("Chunk ", i, " drklgt completed.")

        for paradigm in paradigms:
            posterior[paradigm].append(posterior_chunk[paradigm])
            decoded_pos[paradigm].append(decoded_pos_chunk[paradigm])

    for paradigm in paradigms:
        posterior[paradigm] = np.concatenate(posterior[paradigm], axis=0)
        decoded_pos[paradigm] = np.concatenate(decoded_pos[paradigm], axis=0)
        
    return posterior, decoded_pos


def run_decoder_chance(
        mouse: d.CaimData | d.NpxlData, 
        num_reps: int = 100,
        x: int = 5,
        tunnellength: int = 50, 
        num_pbins: int = 46, 
        SDfrac: float = 0.2, 
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
    triallength_lgt_chunks = u.sort_and_chunk(mouse, mouse.triallength_lgt, 'lgt', discrete, num_chunks)
    triallength_drk_chunks = u.sort_and_chunk(mouse, mouse.triallength_drk, 'drk', discrete, num_chunks)

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
        
        # Smooth data with Gaussian filter
        print("3. Smoothing spikes.")
        sigma = u.compute_sigma(mouse.tau, SDfrac=SDfrac)
        spikes_smoothed = u.gaussiansmooth_spikes(spikes_masked, mask, sigma)

        # Position binning and generating firing rates
        print("4. Position Binning data and generating firing rates.")
        fr_smoothed = u.posbinning_data(
            spikes_smoothed, 
            'spikes', 
            mouse.position_mtx_masked, 
            tunnellength, 
            mouse.tau
        )

        # Split data into light and dark trials
        print("5. Splitting light vs dark.")
        spikes_lgt, spikes_drk = u.split_lightdark(spikes_masked, mouse.darktrials)
        fr_lgt_smoothed, fr_drk_smoothed = u.split_lightdark(fr_smoothed, mouse.darktrials)

        # Scale firing rates
        print("6. Scaling firing rates.")
        fr_lgt_scaled_smoothed, fr_drk_scaled_smoothed = u.scale_firingrate(fr_lgt_smoothed, fr_drk_smoothed)
       
        # Sort and chunk trials in list
        print("Sorting trials and chunking trials.")
        spikes_lgt_chunks = u.sort_and_chunk(mouse, spikes_lgt, 'lgt', discrete, num_chunks)
        spikes_drk_chunks = u.sort_and_chunk(mouse, spikes_drk, 'drk', discrete, num_chunks)
        fr_lgt_smoothed_chunks = u.sort_and_chunk(mouse, fr_lgt_smoothed, 'lgt', discrete, num_chunks)
        fr_drk_smoothed_chunks = u.sort_and_chunk(mouse, fr_drk_smoothed, 'drk', discrete, num_chunks)
        fr_lgt_scaled_smoothed_chunks = u.sort_and_chunk(mouse, fr_lgt_scaled_smoothed, 'lgt', discrete, num_chunks)
        fr_drk_scaled_smoothed_chunks = u.sort_and_chunk(mouse, fr_drk_scaled_smoothed, 'drk', discrete, num_chunks)

        # Intialise output
        paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']
        posterior_rep = {paradigm : [] for paradigm in paradigms}
        decoded_pos_rep = {paradigm : [] for paradigm in paradigms}

        # Decoder training set options
        print("8. Running decoder...")
        training_lgtlgt = fr_lgt_smoothed_chunks
        training_drkdrk = fr_drk_smoothed_chunks
        if scale == True:
            training_lgtdrk = fr_lgt_scaled_smoothed_chunks
            training_drklgt = fr_drk_scaled_smoothed_chunks
        elif scale == False:
            training_lgtdrk = fr_lgt_smoothed_chunks
            training_drklgt = fr_drk_smoothed_chunks

        # Loop through each chunk
        for i in range(num_chunks):
            posterior_chunk = {paradigm: [] for paradigm in paradigms}
            decoded_pos_chunk = {paradigm: [] for paradigm in paradigms}

            print("Decoding rep ", rep, " chunk ", i, "...")
            posterior_chunk['lgtlgt'], decoded_pos_chunk['lgtlgt'] = b.bayesian_decoder_chunks(
                mouse,
                training_lgtlgt[i],
                spikes_lgt_chunks[i],
                triallength_lgt_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "lgtlgt completed.")
            posterior_chunk['drkdrk'], decoded_pos_chunk['drkdrk'] = b.bayesian_decoder_chunks(
                mouse,
                training_drkdrk[i],
                spikes_drk_chunks[i],
                triallength_drk_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "drkdrk completed.")
            posterior_chunk['lgtdrk'], decoded_pos_chunk['lgtdrk'] = b.bayesian_decoder_chunks(
                mouse,
                training_lgtdrk[i],
                spikes_drk_chunks[i],
                triallength_drk_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "lgtdrk completed.")
            posterior_chunk['drklgt'], decoded_pos_chunk['drklgt'] = b.bayesian_decoder_chunks(
                mouse,
                training_drklgt[i],
                spikes_lgt_chunks[i],
                triallength_lgt_chunks[i],
                num_pbins,
                uniformprior
            )
            print("Rep ", rep, "Chunk ", i, "drklgt completed.")

            for paradigm in paradigms:
                posterior_rep[paradigm].append(posterior_chunk[paradigm])
                decoded_pos_rep[paradigm].append(decoded_pos_chunk[paradigm])

        for paradigm in paradigms:
            posterior_rep[paradigm] = np.concatenate(posterior_rep[paradigm], axis=0)
            decoded_pos_rep[paradigm] = np.concatenate(decoded_pos_rep[paradigm], axis=0)
        
        posterior_allreps.append(posterior_rep)
        decoded_pos_allreps.append(decoded_pos_rep)
        print("Completed Rep ", rep)

    print("Completed all reps.")
    
    return posterior_allreps, decoded_pos_allreps


def run_results(
        mouse: d.CaimData | d.NpxlData | d.NpxlData, 
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
        mouse,
        num_chunks: int = 10,
        num_pbins: int = 46,
        discrete: bool = True
):
    """
    """
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']

    # Initialise trial outputs
    trial_accuracy = {paradigm: [] for paradigm in paradigms}
    trial_mean_error = {paradigm: [] for paradigm in paradigms}
    trial_median_error = {paradigm: [] for paradigm in paradigms}
    trial_rt_mse = {paradigm: [] for paradigm in paradigms}
    trial_wt_error = {paradigm: [] for paradigm in paradigms}
    
    # Initialise final outputs
    confusion_mtx = {paradigm: np.zeros((num_pbins, num_pbins)) for paradigm in paradigms}
    MostFreqPred_error = {paradigm: [] for paradigm in paradigms}
    mean_accuracy = {paradigm: [] for paradigm in paradigms}
    mean_error = {paradigm: [] for paradigm in paradigms}
    median_error = {paradigm: [] for paradigm in paradigms}
    rt_mse = {paradigm: [] for paradigm in paradigms}
    mean_wt_error = {paradigm: [] for paradigm in paradigms}
    
    # Sort and chunk true positions
    pos_lgt_sorted = np.concatenate(u.sort_and_chunk(mouse, mouse.pos_lgt_masked, 'lgt', discrete, num_chunks), axis=0)
    pos_drk_sorted = np.concatenate(u.sort_and_chunk(mouse, mouse.pos_drk_masked, 'drk', discrete, num_chunks), axis=0)

    for paradigm in paradigms:
        # Set true position
        if paradigm == 'lgtlgt' or paradigm == 'drklgt':
            true = pos_lgt_sorted
        elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':
            true = pos_drk_sorted

        # Set predicted position
        pred = mouse.decoded_pos[paradigm]
        pred_mask = ~np.isnan(pred)

        # Set posterior
        posterior = mouse.posterior[paradigm]
        num_trials, num_tbins, num_pbins = posterior.shape

        for pos in range(num_pbins):
            # Count frequency for each possible predicted position
            pred_pos_count, num_preds = r.get_pred_pos_count(true, pred, pos)

            # Compute confusion matrix
            for x, count in pred_pos_count.items():
                confusion_mtx[paradigm][int(pos), int(x)] = count / num_preds

            # Compute error for most frequently predicted position
            MostFreqPred_error[paradigm].append(r.get_MostFreqPred_error(pred_pos_count, pos))
        MostFreqPred_error[paradigm] = np.nanmean(MostFreqPred_error[paradigm])

        for trial in range(num_trials):
            # Get trial accuracy, mean_error, median_error, rt_mse, wt_error
            accuracy = r.get_trial_accuracy(true, pred, pred_mask, trial)
            errors = r.get_trial_errors(true, pred, pred_mask, trial)
            wt_error = r.get_trial_wt_error(true, posterior, trial, num_tbins, num_pbins)

            trial_accuracy[paradigm].append(accuracy)
            trial_mean_error[paradigm].append(np.nanmean(errors))
            trial_median_error[paradigm].append(np.nanmedian(errors))
            trial_rt_mse[paradigm].append(np.sqrt(np.nanmean(np.square(errors))))
            trial_wt_error[paradigm].append(wt_error)

        # Average accuracy and errors across trials                                  
        mean_accuracy[paradigm] = np.nanmean(trial_accuracy[paradigm])
        mean_error[paradigm] = np.nanmean(trial_mean_error[paradigm])
        median_error[paradigm] = np.nanmean(trial_median_error[paradigm])
        rt_mse[paradigm] = np.nanmean(trial_rt_mse[paradigm])
        mean_wt_error[paradigm] = np.nanmean(trial_wt_error[paradigm])

    results = {
        'confusion_mtx': confusion_mtx,
        'mean_accuracy': mean_accuracy,
        'mean_error': mean_error,
        'median_error': median_error,
        'rt_mse': rt_mse,
        'mean_wt_error': mean_wt_error,
        'MostFreqPred_error': MostFreqPred_error
    }
    return results


def run_results_chance(
        mouse: d.CaimData, 
        num_reps: int = 100,
        num_chunks: int = 10,
        discrete: bool = True
):
    """
    """
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']
    
    # Initialise final outputs
    mean_accuracy_allreps = []
    mean_error_allreps = []
    median_error_allreps = []
    rt_mse_allreps = []
    mean_wt_error_allreps = []
    MostFreqPred_error_allreps = []

    # Sort and chunk true positions
    pos_lgt_sorted = np.concatenate(u.sort_and_chunk(mouse, mouse.pos_lgt_masked, 'lgt', discrete, num_chunks), axis=0)
    pos_drk_sorted = np.concatenate(u.sort_and_chunk(mouse, mouse.pos_drk_masked, 'drk', discrete, num_chunks), axis=0)

    for rep in range(num_reps):
        # Initialise trial outputs
        trial_accuracy = {paradigm: [] for paradigm in paradigms}
        trial_mean_error = {paradigm: [] for paradigm in paradigms}
        trial_median_error = {paradigm: [] for paradigm in paradigms}
        trial_rt_mse = {paradigm: [] for paradigm in paradigms}
        trial_wt_error = {paradigm: [] for paradigm in paradigms}

        # Intialise rep outputs
        MostFreqPred_error_rep = {paradigm: [] for paradigm in paradigms}
        mean_accuracy_rep = {paradigm: [] for paradigm in paradigms}
        mean_error_rep = {paradigm: [] for paradigm in paradigms}
        median_error_rep = {paradigm: [] for paradigm in paradigms}
        rt_mse_rep = {paradigm: [] for paradigm in paradigms}
        mean_wt_error_rep = {paradigm: [] for paradigm in paradigms}

        for paradigm in paradigms:
            # Set true position
            if paradigm == 'lgtlgt' or paradigm == 'drklgt':
                true = pos_lgt_sorted
            elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':
                true = pos_drk_sorted

            # Set predicted position
            pred = mouse.decoded_pos_allreps[rep][paradigm]
            pred_mask = ~np.isnan(pred)

            # Set posterior
            posterior = mouse.posterior_allreps[rep][paradigm]
            num_trials, num_tbins, num_pbins = posterior.shape

            for pos in range(num_pbins):
                # Count frequency for each possible predicted position
                pred_pos_count, num_preds = r.get_pred_pos_count(true, pred, pos)

                # Compute error for most frequently predicted position
                MostFreqPred_error_rep[paradigm].append(r.get_MostFreqPred_error(pred_pos_count, pos))
            MostFreqPred_error_rep[paradigm] = np.nanmean(MostFreqPred_error_rep[paradigm])

            for trial in range(num_trials):
                # Get trial accuracy, mean_error, median_error, rt_mse, wt_error
                accuracy = r.get_trial_accuracy(true, pred, pred_mask, trial)
                errors = r.get_trial_errors(true, pred, pred_mask, trial)
                wt_error = r.get_trial_wt_error(true, posterior, trial, num_tbins, num_pbins)

                trial_accuracy[paradigm].append(accuracy)
                trial_mean_error[paradigm].append(np.nanmean(errors))
                trial_median_error[paradigm].append(np.nanmedian(errors))
                trial_rt_mse[paradigm].append(np.sqrt(np.nanmean(np.square(errors))))
                trial_wt_error[paradigm].append(wt_error)

            # Average accuracy and errors across trials                                  
            mean_accuracy_rep[paradigm] = np.nanmean(trial_accuracy[paradigm])
            mean_error_rep[paradigm] = np.nanmean(trial_mean_error[paradigm])
            median_error_rep[paradigm] = np.nanmean(trial_median_error[paradigm])
            rt_mse_rep[paradigm] = np.nanmean(trial_rt_mse[paradigm])
            mean_wt_error_rep[paradigm] = np.nanmean(trial_wt_error[paradigm])
        
        # Append rep outputs to final outputs
        mean_accuracy_allreps.append(mean_accuracy_rep)
        mean_error_allreps.append(mean_error_rep)
        median_error_allreps.append(median_error_rep)
        rt_mse_allreps.append(rt_mse_rep)
        mean_wt_error_allreps.append(mean_wt_error_rep)
        MostFreqPred_error_allreps.append(MostFreqPred_error_rep)

    results_allreps = {
        'mean_accuracy_allreps': mean_accuracy_allreps,
        'mean_error_allreps': mean_error_allreps,
        'median_error_allreps': median_error_allreps,
        'rt_mse_allreps': rt_mse_allreps,
        'mean_wt_error_allreps': mean_wt_error_allreps,
        'MostFreqPred_error_allreps': MostFreqPred_error_allreps
    } 
    return results_allreps


def pca_preprocess(
        mouse: d.CaimData,
        rewardzone: list,
        num_chunks: int,
        concat_chunks: bool = True,
        concat_lgtdrk: bool = False
) -> None:
    """
    Preprocess data for PCA analysis.

    Two steps:
    (1) Crop reward zone, trial start and trials with NaN position bins
    (2) Concatenate along trials

    Additional options:
    (1) Concatenate all chunks
    (2) Concatenate light and dark trials
    """
    mouse.fr_lgt_cropped = []
    mouse.fr_drk_cropped = []

    mouse.fr_lgt_processed = []
    mouse.fr_drk_processed = []

    paradigms = ['lgt', 'drk']

    for paradigm in paradigms:
        if paradigm == 'lgt':
            data = mouse.fr_lgt_chunks
        elif paradigm == 'drk':
            data = mouse.fr_drk_chunks

        data_cropped = []
        data_processed = []

        for chunk in range(num_chunks):        
            # Crop reward zone
            # only 46-49 because firing rates are already binned into 50 pbins
            cropped_rz = np.delete(data[chunk], rewardzone, axis=1)

            # Crop trial start NaNs
            # We select the last trial of the chunk because when start positions are
            # continuous in the females (Qiu, Serena), we take the shortest trial after
            # sorting. This is to avoid having NaN position bins at the trial start.
            last_trial = cropped_rz[-1,:,0]
            # Find the last NaN position bin the beginning of the trial and crop
            # everything that's before that bin
            last_nan = np.where(np.isnan(last_trial))[0][-1]
            cropped_start = cropped_rz[:, last_nan +1:, :]
            
            # Crop trials with NaN position bins
            # In some cases, the mouse skips positions and therefore has NaN bins.
            trials_to_remove = np.unique(np.where(np.isnan(cropped_start))[0])
            cropped_nan = np.delete(cropped_start, trials_to_remove, axis=0)
            data_cropped.append(cropped_nan)

            # Concatenate along trials
            data_concatenated = np.concatenate(cropped_nan, axis=0).T
            data_processed.append(data_concatenated)

        if paradigm == 'lgt':
            mouse.fr_lgt_cropped = data_cropped
            mouse.fr_lgt_processed = data_processed
        elif paradigm == 'drk':
            mouse.fr_drk_cropped = data_cropped
            mouse.fr_drk_processed = data_processed

        print(f"PCA pre-processing of {paradigm}:")
        print("      | Cropped             | Concatenated")
        print("chunk | Trial, Pbin, Neuron | Neuron, Trial x Pbin")
        for chunk in range(num_chunks):
            print(chunk, "    |", data_cropped[chunk].shape, "         |", data_processed[chunk].shape)
        print()

    # Concatenate all chunks
    if concat_chunks == True or concat_lgtdrk == True:
        mouse.fr_lgt_concat_chunks = np.concatenate(mouse.fr_lgt_processed, axis=1).T
        mouse.fr_drk_concat_chunks = np.concatenate(mouse.fr_drk_processed, axis=1).T
        print("Concatenated all chunks:")
        print("lgt: ", mouse.fr_lgt_concat_chunks.shape)
        print("drk: ", mouse.fr_drk_concat_chunks.shape)
        print()

    # Concatenate light and dark together
    if concat_lgtdrk == True:
        mouse.fr_lgtdrk_concat = np.concatenate(
            [mouse.fr_lgt_concat_chunks, mouse.fr_drk_concat_chunks], axis=0)
        print("Concatenated light and dark:")
        print("lgtdrk: ", mouse.fr_lgtdrk_concat.shape)


def run_pca(
        data_fit: np.ndarray,
        data_to_transform: np.ndarray = None,
        n_components: int = None
):
    """
    """
    pca = PCA()
    pca.fit(data_fit)
    if data_to_transform is not None:
        if n_components is None:
            data_reduced = pca.transform(data_to_transform)
        else:
            data_reduced = pca.transform(data_to_transform)[:,:n_components]
    else:
        if n_components is None:
            data_reduced = pca.transform(data_fit)
        else:
            data_reduced = pca.transform(data_fit)[:,:n_components]

    csum = np.cumsum(pca.explained_variance_ratio_)

    print("Original data shape:")
    print(data_fit.shape)
    print()
    print("Reduced data shape:")
    print(data_reduced.shape)
    print()
    print(f"Explained variance of the first {n_components} components:")
    print(pca.explained_variance_ratio_[:n_components])
    print()
    print("num of components explaining 50% Var:", np.where(csum > 0.5)[0][0] +1)
    print("num of components explaining 70% Var:", np.where(csum > 0.7)[0][0] +1)
    print("num of components explaining 90% Var:", np.where(csum > 0.9)[0][0] +1)

    plt.plot(csum)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    
    return pca, data_reduced


def pca_postprocess(mouse, num_chunks, n_components):
    """
    """

    if getattr(mouse, 'fr_lgtdrk_reduced', None) is not None:
        # Split light and dark components
        data_reduced_lgt, data_reduced_drk = np.split(
            mouse.fr_lgtdrk_reduced, [mouse.fr_lgt_concat_chunks.shape[0]], axis=0)
        datasets = {"lgt": data_reduced_lgt, "drk": data_reduced_drk}
        print("Post-processing mouse.fr_lgtdrk_reduced")
    else:
        datasets = {"lgt": mouse.fr_lgt_reduced, "drk": mouse.fr_drk_reduced}
        print("Post-processing mouse.fr_lgt_reduced and mouse.fr_drk_reduced")

    output = {}
    
    for paradigm, data in datasets.items():
        # Find indices to split data
        chunk_sizes = [mouse.fr_lgt_processed[i].shape[1] if paradigm == "lgt"
                       else mouse.fr_drk_processed[i].shape[1] for i in range(num_chunks)]
        split_indices = np.cumsum(chunk_sizes[:-1])

        print(f"Chunk sizes ({paradigm}):", chunk_sizes)

        # Split reduced data into chunks
        data_split = np.split(data, split_indices, axis=0)
        data_cropped = getattr(mouse, f"fr_{paradigm}_cropped")

        # Reshape data into Trials x Pbins
        data_reshaped = []
        for i in range(num_chunks):
            chunk_reshaped = data_split[i].reshape(
                data_cropped[i].shape[0],
                data_cropped[i].shape[1],
                n_components
            )
            data_reshaped.append(chunk_reshaped)

        # Pad NaNs to trial start and rewardzone
        data_padded = []
        max_pbins = max(chunk.shape[1] for chunk in data_reshaped)
        for chunk in data_reshaped:
            # Find diff between the len of chunk and the longest chunk
            pad_size = max_pbins - chunk.shape[1]
            # Add NaNs to first 5 bins (trial start) and last 4 bins (rewardzone)
            pad_width = [(0, 0), (pad_size+5, 4), (0, 0)]
            chunk_padded = np.pad(chunk, pad_width, mode='constant', constant_values=np.nan)
            data_padded.append(chunk_padded)
   
        output[paradigm] = data_padded

        print(f"PCA post-processing of {paradigm}:")
        print("      | Before             | Reshaped")
        print("chunk | Trial x Pbin, Component | Trial, Pbin, Component")
        for chunk in range(num_chunks):
            print(chunk, "    |", data_split[chunk].shape, "         |", data_padded[chunk].shape)
        print()

    mouse.fr_lgt_reconstructed = output['lgt']
    mouse.fr_drk_reconstructed = output['drk']
