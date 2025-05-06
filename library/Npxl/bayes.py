import sys
import os
import numpy as np
from scipy.special import factorial

sys.path.append(os.path.abspath('../library/Npxl'))
import data as d
import utils_npxl as u

def bayesian_decoder(
        mouse: d.CaimData | d.NpxlData, 
        training_fr: np.ndarray, 
        testing_spikes: np.ndarray, 
        num_pbins: int,
        uniform_prior: bool = False
    ) -> tuple:
    """
    Bayesian decoder.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's info to use.

        training_fr (np.ndarray):
            (Trial, Position Bin, Neuron) spatial tuning for the decoder to learn.

        testing_spikes (np.ndarray):
            (Trial, Time bin, Neuron) spikes data to test the decoder.

        num_pbins (int):
            Number of postion bins to decode, excluding the reward zone.

        uniform_prior (bool):
            Default false, p(x) is variable depending on trial length of each trial.
            If true, p(x) = 1 / num_pbins.

    Returns:
        tuple: a tuple containing:
            - posterior (np.ndarray):
                (Trial, Time Bin, Position Bin) posterior probability computed.
            - decoded_pos (np.ndarray):
                (Trial, Time Bin) decoded positions derived from max posterior.
    """
    num_trials, num_tbins, num_neurons = testing_spikes.shape    
    # Initialise ouput matrices
    posterior = np.full((num_trials, num_tbins, num_pbins), np.nan)   
    decoded_pos = np.full((num_trials, num_tbins), np.nan)    

    if uniform_prior is True:
        px = 1 / num_pbins
        log_px = np.log(px)
    else: pass
      
    # Run through equation by trial
    for test_trial in range(num_trials):

        # Remove test trial from training set if both training and testing set
        # are from the same condition, i.e. both light or both dark.
        if training_fr.shape[0] == num_trials:
            firingrate = np.delete(training_fr, test_trial, axis=0)
        else:
            firingrate = training_fr

        # ----------------------------------------------------------------------
        # Compute log p(x) and p(x)
        # ----------------------------------------------------------------------
        if uniform_prior is False:        
            # Set p(x) = 1 over trial length measured in position bins
            if num_trials == mouse.triallength_lgt.shape[0]:
                px = 1 / mouse.triallength_lgt[test_trial]
            elif num_trials == mouse.triallength_drk.shape[0]:
                px = 1 / mouse.triallength_drk[test_trial]
            else:
                print(num_trials)
                print(mouse.triallength_lgt.shape[0])
                print(mouse.triallength_drk.shape[0])
                raise ValueError("num_trials does not match the shape of either trial_light or trial_dark.")
            log_px = np.log(px)
        else: pass
                  
        # ----------------------------------------------------------------------
        # Set equation variables
        # ----------------------------------------------------------------------
        # (Time Bin, Neuron)  
        spikecount = testing_spikes[test_trial]    # spike count of test trial
        nfac = factorial(spikecount, exact=True)   # n!
        log_nfac = np.log(nfac.astype(float))      # log(n!)

        # (Position Bin, Neuron)
        fx = np.nanmean(firingrate, axis=0)        # mean firing rate across trials
        lamb = mouse.tau * fx                      # lambda or mean spike count
        log_lamb = np.log(lamb)                    # log lambda
 
        # ----------------------------------------------------------------------
        # Compute log p(n|x) and p(n|x)
        # ----------------------------------------------------------------------
        # Intialise log p(n|x) before and after sum across neurons
        # (Time Bin, Position Bin, Neuron)
        log_pnx_before_sum = np.full((num_tbins, num_pbins, num_neurons), -np.inf)
        # (Time Bin, Position Bin)
        log_pnx = np.full((num_tbins, num_pbins), -np.inf)
        
        # Main equation
        for x in range(num_pbins):
            for i in range(num_neurons):
                for t in range(num_tbins):
                    n_i = spikecount[t,i]                
                    log_pnx_before_sum[t,x,i] = (n_i * log_lamb[x,i]) - lamb[x,i] - log_nfac[t,i]        
        
        # Check for NaNs before sum across neurons
        for x in range(num_pbins):
            for t in range(num_tbins):
                # Get log p(n|x) values of all neurons in this time bin and position bin
                all_neuron_values = log_pnx_before_sum[t, x]
                # If all are NaNs, skip.
                if np.all(np.isnan(all_neuron_values)):
                    continue # log_pnx remain NaNs
                else:
                # Sum across neurons for this time bin and position bin
                    log_pnx[t, x] = np.nansum(all_neuron_values)
        
        pnx = np.exp(log_pnx)
        
        # ----------------------------------------------------------------------
        # Compute log p(n) for each time bin
        # ----------------------------------------------------------------------
        # (Time Bin)              
        log_pn = np.log(np.sum((pnx * px), axis=1))       

        # ----------------------------------------------------------------------
        # Compute posterior p(x|n) for each time bin and position bin
        # ----------------------------------------------------------------------      
        # (Time Bin, Position Bin)
        posterior[test_trial] = np.exp((log_pnx.T + log_px - log_pn).T)
        
        # ----------------------------------------------------------------------
        # Get decoded positions from posterior
        # ----------------------------------------------------------------------
        for tbin in range(num_tbins):            
            # if max posterior == min posterior, decoded_pos remains NaN
            if np.nanmax(posterior[test_trial,tbin]) == np.nanmin(posterior[test_trial,tbin]):
                decoded_pos[test_trial,tbin] = np.nan
            # if posterior all NaNs and thus max posterior is NaN, decoded_pos remains NaN    
            elif np.isnan(np.nanmax(posterior[test_trial,tbin])):
                decoded_pos[test_trial,tbin] = np.nan
            else:
                decoded_pos[test_trial,tbin] = np.argmax(posterior[test_trial,tbin])

        # ----------------------------------------------------------------------
        # Repeat for the next trial
        
    return posterior, decoded_pos



def bayesian_decoder_chunks(
        mouse: d.CaimData | d.NpxlData, 
        training_fr_chunk: np.ndarray, 
        testing_spikes_chunk: np.ndarray,
        trial_length_chunk: np.ndarray,
        num_pbins: int,
        uniform_prior: bool = False,
    ) -> tuple:
    """
    Bayesian decoder for chunked trials. Pass only data of each chunk.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's info to use.

        training_fr (np.ndarray):
            (Trial, Position Bin, Neuron) spatial tuning for the decoder to learn.

        testing_spikes (np.ndarray):
            (Trial, Time bin, Neuron) spikes data to test the decoder.

        num_pbins (int):
            Number of postion bins to decode, excluding the reward zone.

        uniform_prior (bool):
            Default false, p(x) is variable depending on trial length of each trial.
            If true, p(x) = 1 / num_pbins.

        trial_length (np.ndarray):
            Trial length of each trial in the chunk.

    Returns:
        tuple: a tuple containing:
            - posterior (np.ndarray):
                (Trial, Time Bin, Position Bin) posterior probability computed.
            - decoded_pos (np.ndarray):
                (Trial, Time Bin) decoded positions derived from max posterior.
    """
    num_trials, num_tbins, num_neurons = testing_spikes_chunk.shape    
    # Initialise ouput matrices
    posterior_chunk = np.full((num_trials, num_tbins, num_pbins), np.nan)   
    decoded_pos_chunk = np.full((num_trials, num_tbins), np.nan)

    # Compute log p(x) and p(x) for uniform prior
    if uniform_prior is True:
        px = 1 / num_pbins
        log_px = np.log(px)
    else: pass
      
    # Run through equation by trial
    for test_trial in range(num_trials):

        # Remove test trial from training set if both training and testing set
        # are from the same condition, i.e. both light or both dark.
        if training_fr_chunk.shape[0] == num_trials:
            firingrate = np.delete(training_fr_chunk, test_trial, axis=0)
        else:
            firingrate = training_fr_chunk

        # ----------------------------------------------------------------------
        # Compute log p(x) and p(x)
        # ----------------------------------------------------------------------
        if uniform_prior is False:        
            # Set p(x) = 1 over trial length measured in position bins
            px = 1 / trial_length_chunk[test_trial]
            log_px = np.log(px)
        else: pass
                  
        # ----------------------------------------------------------------------
        # Set equation variables
        # ----------------------------------------------------------------------
        # (Time Bin, Neuron)  
        spikecount = testing_spikes_chunk[test_trial]   # spike count of test trial
        nfac = factorial(spikecount, exact=True)        # n!
        log_nfac = np.log(nfac.astype(float))           # log(n!)

        # (Position Bin, Neuron)
        fx = np.nanmean(firingrate, axis=0)        # mean firing rate across trials
        lamb = mouse.tau * fx                      # lambda or mean spike count
        log_lamb = np.log(lamb)                    # log lambda
 
        # ----------------------------------------------------------------------
        # Compute log p(n|x) and p(n|x)
        # ----------------------------------------------------------------------
        # Intialise log p(n|x) before and after sum across neurons
        # (Time Bin, Position Bin, Neuron)
        log_pnx_before_sum = np.full((num_tbins, num_pbins, num_neurons), -np.inf)
        # (Time Bin, Position Bin)
        log_pnx = np.full((num_tbins, num_pbins), -np.inf)
        
        # Main equation
        for x in range(num_pbins):
            for i in range(num_neurons):
                for t in range(num_tbins):
                    n_i = spikecount[t,i]                
                    log_pnx_before_sum[t,x,i] = (n_i * log_lamb[x,i]) - lamb[x,i] - log_nfac[t,i]        
        
        # Check for NaNs before sum across neurons
        for x in range(num_pbins):
            for t in range(num_tbins):
                # Get log p(n|x) values of all neurons in this time bin and position bin
                all_neuron_values = log_pnx_before_sum[t, x]
                # If all are NaNs, skip.
                if np.all(np.isnan(all_neuron_values)):
                    continue # log_pnx remain NaNs
                else:
                # Sum across neurons for this time bin and position bin
                    log_pnx[t, x] = np.nansum(all_neuron_values)
        
        pnx = np.exp(log_pnx)
        
        # ----------------------------------------------------------------------
        # Compute log p(n) for each time bin
        # ----------------------------------------------------------------------
        # (Time Bin)              
        log_pn = np.log(np.sum((pnx * px), axis=1))       

        # ----------------------------------------------------------------------
        # Compute posterior p(x|n) for each time bin and position bin
        # ----------------------------------------------------------------------      
        # (Time Bin, Position Bin)
        posterior_chunk[test_trial] = np.exp((log_pnx.T + log_px - log_pn).T)
        
        # ----------------------------------------------------------------------
        # Get decoded positions from posterior
        # ----------------------------------------------------------------------
        for tbin in range(num_tbins):            
            # if max posterior == min posterior, decoded_pos remains NaN
            if np.nanmax(posterior_chunk[test_trial,tbin]) == np.nanmin(posterior_chunk[test_trial,tbin]):
                decoded_pos_chunk[test_trial,tbin] = np.nan
            # if posterior all NaNs and thus max posterior is NaN, decoded_pos remains NaN    
            elif np.isnan(np.nanmax(posterior_chunk[test_trial,tbin])):
                decoded_pos_chunk[test_trial,tbin] = np.nan
            else:
                decoded_pos_chunk[test_trial,tbin] = np.argmax(posterior_chunk[test_trial,tbin])

        # ----------------------------------------------------------------------
        # Repeat for the next trial
        
    return posterior_chunk, decoded_pos_chunk


