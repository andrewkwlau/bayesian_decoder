import sys
import os
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath('../library/Npxl'))


def get_trial_accuracy(true, pred, pred_mask, trial):
    """
    Args:
        true:
            true positions of a given trial
        
        pred:
            predicted positions of a given trial

        pred_mask:
            mask of predicted positions of a given trial excluding NaNs

    Returns:
        accuracy_rate:
            accuracy rate of the trial
    """
    # Array of True/False for correct/incorrect predictions
    check_equal = pred[trial][pred_mask[trial]] == true[trial][pred_mask[trial]]
    # Sum of True values divided by total number of predictions
    accuracy_rate = np.sum(check_equal) / len(check_equal)

    return accuracy_rate


def get_trial_errors(true, pred, pred_mask, trial):
    """
    Args:
        true:
            true positions of a given trial
        
        pred:
            predicted positions of a given trial

        pred_mask:
            mask of predicted positions of a given trial excluding NaNs

    Returns:
        errors:
            absolute errors of the trial
    """
    # Absoulte errors
    errors = np.abs(pred[trial][pred_mask[trial]] - true[trial][pred_mask[trial]])

    return errors


def get_trial_wt_error(true, posterior, trial, num_tbins, num_pbins):
    """
    Args:
        true:
            true positions of a given trial
        
        posterior:
            posterior probabilities of a given trial

        num_tbins:
            number of time bins

        num_pbins:
            number of position bins

    Returns:
        mean_wt_error:
            mean weighted error of the trial
    """
    wt_errors_sum = []

    # For all time bins of the trial
    for tbin in range(num_tbins):
        wt_errors = []

        # If true position is not NaN
        if np.isnan(true[trial, tbin]):
                continue
        else:
            # Compute weighted errors for all possible position bins
            for pbin in range(num_pbins):
                wt_errors.append(abs((true[trial, tbin] - pbin) * posterior[trial, tbin, pbin]))
    
        if np.all(np.isnan(wt_errors)):
            wt_errors_sum.append(np.nan)
        else:
            # Sum weighted errors across all position bins
            wt_errors_sum.append(np.nansum(wt_errors))

    # Compute mean weighted error
    mean_wt_error = np.nanmean(wt_errors_sum)

    return mean_wt_error


def get_pred_pos_count(true, pred, pos):
    """
    Args:
        true:
            true positions of the whole session
        
        pred:
            predicted positions of the whole session

        pos:
            true position bin index

    Returns:
        pred_pos_count:
            Counter object of predicted positions for given true positions

        num_pred:
            total number of predictions for givne true position
    """

    # Find time bins where true position is pos
    true_pos_tbins = list(zip(*np.where(true == pos)))
    # Find corresponding predicted positions in those time bins excluding NaNs
    predictions = [pred[i] for i in true_pos_tbins if not np.isnan(pred[i])]

    # Get frequency of each predicted position and total number of predctions
    num_preds = len(predictions)
    pred_pos_count = Counter(predictions)

    return pred_pos_count, num_preds
    

def get_MostFreqPred_error(pred_pos_count, pos):
    """
    Finds most frequently decoded positions (multiple) for given true position
    and compute mean error against true position.

    Args:
        pred_pos_count:
            Counter object of predicted positions for given true positions

        pos:
            true position bin index

    Returns:
        MostFreqPred_error:
            Mean error of the most frequent predicted positions for given true position
    """
    # If pred_pos_count is empty, return NaN
    if not pred_pos_count:
        return np.nan
    else:
        # Find the highest count in pred_pos_count
        max_count = max(pred_pos_count.values())

        # Find the predicted positions (multiple) with the highest count
        most_freq_preds = [pred for pred, count in pred_pos_count.items() if count == max_count]

        # Compute error between given true position and each of the most frequent predicted positions
        errors = [abs(pred - pos) for pred in most_freq_preds]
        
        # Compute mean error
        MostFreqPred_error = np.nanmean(errors)

        return MostFreqPred_error