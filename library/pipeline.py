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
        mouse: d.CaImgData,
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
    mouse.triallength_lgt, mouse.triallength_drk = u.split_lightdark(mouse.triallength, mouse.darktrials)

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


def run_decoder(
        mouse: d.CaImgData | d.NpxlData | d.NpxlData,
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
    if type(mouse) == d.CaImgData | d.NpxlData:
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
        mouse: d.CaImgData | d.NpxlData | d.NpxlData, 
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
    if type(mouse) == d.CaImgData | d.NpxlData:
        get_tuning_curves(mouse, x, tunnellength, SDfrac)
    elif type(mouse) == d.NpxlData:
        get_tuning_curves_npxl(mouse, x, tunnellength)

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

    num_chunks = len(spikes_lgt_chunks)

    # Intialise output
    posterior_allchunks = []
    decoded_pos_allchunks = []

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
        mouse: d.CaImgData | d.NpxlData, 
        num_reps: int,
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
        sigma = u.compute_sigma(mouse.tau, SDfrac=SDfrac)
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
        mouse: d.CaImgData | d.NpxlData | d.NpxlData, 
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
        mouse: d.CaImgData | d.NpxlData, 
        num_pbins: int = 46,
        num_chunks: int = 10,
        discrete: bool = True
) -> tuple:
    """
    Run confusion matrices, accuracy, and errors.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        num_pbins (int):
            Number of position bins to generate confusion matrix (excluding the reward zone).

        num_chunks (int):
            Number of chunks of trials.

        discrete: (bool):
            Whether to chunk trials in discrete or continuous manner.

    Returns:
        dict: a dict containing:
            - confusion_mtx_allchunks (dict):
                a dict containing the confusion matrices (np.ndarray) of each paradigm

            - mean_accuracy (dict):
                a dict containing the mean accuracy (float) across chunks of each paradigm

            - mean_error (dict):
                a dict containing the mean error (flaot) across chunks of each paradigm

            - median_error (dict):
                a dict containing the median error (float) across chunks of each paradigm

            - rt_mse (dict):
                a dict containing the root mean squared error (float) across chunks of each paradigm

            - most_freq_mean_error (dict):
                a dict containing the mean error (float) of the most frequently decoded position of each paradigm

    """  
    # Chunk position matrix
    mouse.pos_lgt_chunks = u.sort_and_chunk(mouse, mouse.pos_lgt_masked, 'lgt', discrete, num_chunks)
    mouse.pos_drk_chunks = u.sort_and_chunk(mouse, mouse.pos_drk_masked, 'drk', discrete, num_chunks)

    # Intialise output
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']
    confusion_mtx_allchunks = {paradigm: np.zeros((num_pbins, num_pbins)) for paradigm in paradigms}
    accuracy_mtx = {paradigm: [] for paradigm in paradigms}
    mean_accuracy = {paradigm: [] for paradigm in paradigms}
    errors = {paradigm: [] for paradigm in paradigms}
    mean_error = {paradigm: [] for paradigm in paradigms}
    median_error = {paradigm: [] for paradigm in paradigms}
    rt_mse = {paradigm: [] for paradigm in paradigms}
    all_predictions = {paradigm: {pos: [] for pos in range(num_pbins)} for paradigm in paradigms}
    most_freq_predictions = {paradigm: {pos: [] for pos in range(num_pbins)} for paradigm in paradigms}
    most_freq_errors = {paradigm: {pos: [] for pos in range(num_pbins)} for paradigm in paradigms}
    most_freq_mean_error = {paradigm: [] for paradigm in paradigms}

    # Loop through each paradigm and chunk
    for paradigm in paradigms:
        for chunk in range(num_chunks):
            # Set true position
            if paradigm == 'lgtlgt' or paradigm == 'drklgt':
                true = mouse.pos_lgt_chunks[chunk]
            elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':
                true = mouse.pos_drk_chunks[chunk]
            # Set predicted position and create mask for time bins where there are predictions
            pred = mouse.decoded_pos_allchunks[chunk][paradigm]
            pred_mask = ~np.isnan(pred)

            # True/False matrix of pred pos == true pos, where there are decoder predictions
            accuracy_mtx[paradigm].extend(pred[pred_mask] == true[pred_mask])

            # Compute error for each time bin where there are decoder predictions
            errors[paradigm].extend(abs(np.subtract(pred[pred_mask], true[pred_mask])))

            # For each true position bin y
            for pos in range(num_pbins):        
                # Find time bins where true pos is y
                true_pos_tbins = list(zip(*np.where(true == pos)))
                # Find corresponding predicted positions in those time bins excluding NaNs
                predictions = [pred[i] for i in true_pos_tbins if not np.isnan(pred[i])]
                all_predictions[paradigm][pos].extend(predictions)

                # Get frequency of each predicted pos and compute % of total predictions for confusion matrix
                total_num_predictions = len(all_predictions[paradigm][pos])
                prediction_count_for_pos = Counter(all_predictions[paradigm][pos])
                for x, count in prediction_count_for_pos.items():
                    confusion_mtx_allchunks[paradigm][int(pos), int(x)] = count / total_num_predictions

                # Find most frequently decoded position for true position y and their respective errors
                if not prediction_count_for_pos:
                    most_freq_predictions[paradigm][pos] = np.nan
                else:
                    max_count = max(prediction_count_for_pos.values())
                    most_freq_predictions[paradigm][pos] = [pred for pred, count in prediction_count_for_pos.items() if count == max_count]
                    most_freq_errors[paradigm][pos] = [abs(pos - pred) for pred in most_freq_predictions[paradigm][pos]]
                    most_freq_errors[paradigm][pos] = np.nanmean(most_freq_errors[paradigm][pos])
        
        # Compute accuracy rate
        mean_accuracy[paradigm] = np.sum(accuracy_mtx[paradigm]) / len(accuracy_mtx[paradigm])

        # Compute mean error, median error and root MSE
        mean_error[paradigm] = np.nanmean(errors[paradigm])
        median_error[paradigm] = np.nanmedian(errors[paradigm])
        rt_mse[paradigm] = np.sqrt(np.nanmean(np.square(errors[paradigm])))

        # Compute mean error for most frequently decoded position
        most_freq_errors[paradigm] = np.array([v for v in most_freq_errors[paradigm].values() if isinstance(v, (int, float))])
        most_freq_mean_error[paradigm] = np.nanmean(most_freq_errors[paradigm])

    results = {
        'confusion_mtx': confusion_mtx_allchunks,
        'mean_accuracy': mean_accuracy,
        'mean_error': mean_error,
        'median_error': median_error,
        'rt_mse': rt_mse,
        'most_freq_mean_error': most_freq_mean_error
    }  
    return results


def run_results_chance(
        mouse: d.CaImgData | d.NpxlData, 
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


def pca_preprocess(
        mouse: d.CaImgData,
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
