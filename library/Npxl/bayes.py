import numpy as np
from scipy.special import factorial, logsumexp, gammaln

import data as d


def bayes(
    data, # data object e.g. d.Data
    fr_tuning_curve: np.ndarray,
    spikes: np.ndarray,
    trial_length: np.ndarray,
    num_pbins: int,
    paradigm: str
):
    """
    """
    num_trials, num_tbins, num_units = spikes.shape
    
    # Slice rewardzone from tuning curve
    fr_tuning_curve = fr_tuning_curve[:, :num_pbins, :]

    # Initialise output
    posterior = np.full((num_trials, num_tbins, num_pbins), np.nan)
    decoded_pos = np.full((num_trials, num_tbins), np.nan)

    for test_trial in range(num_trials):
        # Remove test trial from training data
        if paradigm == 'lgtlgt' or paradigm == 'drkdrk':
            tuning = np.delete(fr_tuning_curve, test_trial, axis=0)
        else:
            tuning = fr_tuning_curve

        # ----------------------------------------------------------------------
        # Initialise equation variables
        # ----------------------------------------------------------------------
        epsilon = 1e-10  # small constant to avoid log(0)
        # Shape: Tbin, Units
        n = spikes[test_trial]                          # spike count of test trial (n)
        # log_nfac = np.log(factorial(n))                 # log(n!)
        log_nfac = gammaln(n + 1)                       # more stable for large n
        # Shape: Pbin, Units
        fx = np.nanmean(tuning, axis=0)                 # mean firingrate across trials
        lamb = data.tau * fx                            # lambda or mean spike count
        log_lamb = np.log(lamb + epsilon)               # log(lambda)

        # ----------------------------------------------------------------------
        # Compute log p(n|x)
        # ----------------------------------------------------------------------
        sum_of_terms = np.full((num_tbins, num_pbins, num_units), -np.inf)
        log_pnx = np.full((num_tbins, num_pbins), -np.inf)
        
        # Shape: Tbin, Pbin, Units
        term1 = n[:, np.newaxis, :] * log_lamb[np.newaxis, :, :]    # n * log(lambda)
        term2 = -lamb[np.newaxis, :, :]                             # -lambda
        term3 = -log_nfac[:, np.newaxis, :]                         # -log(n!)
        sum_of_terms = term1 + term2 + term3
  
        # Sum over units; Shape: Tbin, Pbin
        all_nan_mask = np.all(np.isnan(sum_of_terms), axis=2)
        log_pnx_sum = np.nansum(sum_of_terms, axis=2)  # nans become 0s, so we’ll fix below
        log_pnx = np.where(all_nan_mask, -np.inf, log_pnx_sum)

        # ----------------------------------------------------------------------
        # Compute log p(x) and log p(n)
        # ----------------------------------------------------------------------
        # Shape: Pbin
        px = 1 / trial_length[test_trial]
        log_px = np.log(px)                             # log p(x)
        # Shape: Tbin
        log_pn = logsumexp(log_pnx + log_px, axis=1)    # log p(n)

        # ----------------------------------------------------------------------
        # Compute posterior p(x|n) and decoded position
        # ----------------------------------------------------------------------
        # Shape: Tbin, Pbin
        log_pxn = log_pnx + log_px - log_pn[:, np.newaxis]
        pxn = np.exp(log_pxn)
        posterior[test_trial] = pxn

        for tbin in range(num_tbins):
            if np.isnan(np.nanmax(posterior[test_trial, tbin])):
                continue
            elif np.nanmax(posterior[test_trial, tbin]) == np.nanmin(posterior[test_trial, tbin]):
                continue
            else:
                # max_pxn = np.nanmax(pxn[tbin])
                # max_pbins = np.where(np.isclose(pxn[tbin], max_pxn))[0]
                # decoded_pos[test_trial, tbin] = np.random.choice(max_pbins)
                decoded_pos[test_trial, tbin] = np.nanargmax(posterior[test_trial, tbin])


        # if paradigm == 'lgtlgt':
        #     print("-" * 40)
        #     print("Paradigm:", paradigm)
        #     print("Trial", test_trial)
        #     print("fx nan?", np.isnan(fx).all())
        #     print("fx zero?", np.all(fx == 0))
        #     print(f"Max Spike Count (n): {np.nanmax(n)}")
        #     print(f"Unique log_nfac values: {np.unique(log_nfac)}")
        #     print(f"Number of 'inf' log_nfac values: {np.sum(np.isinf(log_nfac))}")
        #     print("lamb nan?", np.isnan(lamb).all())
        #     print("log_lamb nan?", np.isnan(log_lamb).all())
        #     print("log_px:", log_px)
        #     print("unique log_pnx:", np.unique(log_pnx))
        #     print("log_pn:", log_pn)


        #     unit_zero_everywhere = np.all(tuning == 0, axis=(0,1))
        #     print("Units with zero tuning curve in ALL bins:", np.where(unit_zero_everywhere)[0])
        #     bad_units = np.any(fx == 0, axis=0)   # shape: (num_units,)
        #     print("Units with zero tuning curves:", np.where(bad_units)[0])
        #     bad_units = np.any(np.isinf(log_lamb), axis=0)
        #     print("Units with log_lamb == -inf:", np.where(bad_units)[0])
        #     bad_units = np.any(np.isinf(sum_of_terms), axis=(0,1))
        #     print("Units causing -inf in the likelihood:", np.where(bad_units)[0])


    return posterior, decoded_pos