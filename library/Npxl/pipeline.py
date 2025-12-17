import numpy as np
import pandas as pd

import data as d
import utils as u
import bayes as b
import results as r
import figures as figs


def get_tuning_curves(
        data, # data object e.g. d.Data
        x: int = 5,
        num_pbins: int = 50,
        rel_pos: bool = False
):
    """
    Get tuning curves for light and dark conditions, masking out light sections
    """
    # Find trial start locations and trial length
    data.trial_startloc_lgt = u.find_trial_startloc(data, data.pos_lgt)
    data.trial_startloc_drk = u.find_trial_startloc(data, data.pos_drk)

    # Find light sections of the trial 
    data.light_section_lgt = u.get_light_section(data.pos_lgt, data.trial_startloc_lgt, x)
    data.light_section_drk = u.get_light_section(data.pos_drk, data.trial_startloc_drk, x)

    # Mask the light section and reward zone in data
    data.mask_lgt = u.create_mask(data.pos_lgt, data.light_section_lgt, data.rewardzone)
    data.mask_drk = u.create_mask(data.pos_drk, data.light_section_drk, data.rewardzone)
    data.fr_lgt_masked = u.apply_mask(data.fr_lgt, data.mask_lgt)
    data.fr_drk_masked = u.apply_mask(data.fr_drk, data.mask_drk)
    data.pos_lgt_masked = u.apply_mask(data.pos_lgt, data.mask_lgt)
    data.pos_drk_masked = u.apply_mask(data.pos_drk, data.mask_drk)

    if rel_pos == False:
        print("\nGetting tuning curves by absolute position ...")
        # Get trial lengths
        data.trial_length_lgt = u.get_trial_length(data.pos_lgt_masked)
        data.trial_length_drk = u.get_trial_length(data.pos_drk_masked)
        # Get tuning curves by binning data by positions
        data.tuning_curves_lgt = u.pos_binning(data.fr_lgt_masked, data.pos_lgt_masked, num_pbins)
        data.tuning_curves_drk = u.pos_binning(data.fr_drk_masked, data.pos_drk_masked, num_pbins)
    else:
        print("\nGetting tuning curves by relative position ...")
        data.trial_length_lgt = np.ones(data.pos_lgt_masked.shape[0])
        data.trial_length_drk = np.ones(data.pos_drk_masked.shape[0])
        # Get relative positions
        data.relpos_lgt = u.get_relative_position(data.pos_lgt_masked, num_pbins)
        data.relpos_drk = u.get_relative_position(data.pos_drk_masked, num_pbins)
        # Get relative position tuning curves
        data.tuning_curves_lgt = u.pos_binning(data.fr_lgt_masked, data.relpos_lgt, num_pbins)
        data.tuning_curves_drk = u.pos_binning(data.fr_drk_masked, data.relpos_drk, num_pbins)

    print(data.tuning_curves_lgt.shape)
    print(data.tuning_curves_drk.shape)





def decode_positions(
        data, # data object e.g. d.Data
        x: int = 5,
        num_pbins: int = 50,
        num_pbins_decode: int = 46,
        rel_pos: bool = False,
        grouping: bool = True
):
    """
    """
    # Get tuning curves
    get_tuning_curves(data, x, num_pbins, rel_pos)

    # Initalise output
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']
    POSTERIOR = {paradigm: [] for paradigm in paradigms}
    DECODED_POS = {paradigm: [] for paradigm in paradigms}

    # Set parameters for decoder function
    if grouping == True:
        # Sort and group trials
        print("\nSorting and grouping trials by start location ...")
        data.spikes_lgt_groups = u.sort_and_group(data.spikes_lgt, data.trial_startloc_lgt)
        data.spikes_drk_groups = u.sort_and_group(data.spikes_drk, data.trial_startloc_drk)
        data.tuning_curves_lgt_groups = u.sort_and_group(data.tuning_curves_lgt, data.trial_startloc_lgt)
        data.tuning_curves_drk_groups = u.sort_and_group(data.tuning_curves_drk, data.trial_startloc_drk)
        data.trial_length_lgt_groups = u.sort_and_group(data.trial_length_lgt, data.trial_startloc_lgt)
        data.trial_length_drk_groups = u.sort_and_group(data.trial_length_drk, data.trial_startloc_drk)

        print("\nDecoding with grouping by start location ...")
        print("Number of start location groups:", len(data.spikes_lgt_groups))
        params = {
            'lgtlgt': (data.tuning_curves_lgt_groups, data.spikes_lgt_groups, data.trial_length_lgt_groups),
            'drkdrk': (data.tuning_curves_drk_groups, data.spikes_drk_groups, data.trial_length_drk_groups),
            'lgtdrk': (data.tuning_curves_lgt_groups, data.spikes_drk_groups, data.trial_length_drk_groups),
            'drklgt': (data.tuning_curves_drk_groups, data.spikes_lgt_groups, data.trial_length_lgt_groups)
        }
    else:
        print("\nDecoding without grouping by start location ...")
        params = {
            'lgtlgt': ([data.tuning_curves_lgt], [data.spikes_lgt], [data.trial_length_lgt]),
            'drkdrk': ([data.tuning_curves_drk], [data.spikes_drk], [data.trial_length_drk]),
            'lgtdrk': ([data.tuning_curves_lgt], [data.spikes_drk], [data.trial_length_drk]),
            'drklgt': ([data.tuning_curves_drk], [data.spikes_lgt], [data.trial_length_lgt])
        }

    # Decode by paradigm and start location group
    for paradigm in paradigms:
        tuning_curve, spikes, trial_length = params[paradigm]

        for i in range(len(spikes)):
            posterior, decoded_pos = b.bayes(
                data, tuning_curve[i], spikes[i], trial_length[i], num_pbins_decode, paradigm
            )
            POSTERIOR[paradigm].append(posterior)
            DECODED_POS[paradigm].append(decoded_pos)

        print(f"{paradigm} completed.")
        # Stitch the trial groups back together
        POSTERIOR[paradigm] = np.concatenate(POSTERIOR[paradigm], axis=0)
        DECODED_POS[paradigm] = np.concatenate(DECODED_POS[paradigm], axis=0)

    return POSTERIOR, DECODED_POS




def get_decoding_results(
        data, # data object e.g. d.Data
        num_pbins: int = 46,
        rel_pos: bool = False
):
    """
    """
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']

    # Initialise output
    results = {
        "confusion_mtx": {paradigm: np.zeros((num_pbins, num_pbins)) for paradigm in paradigms},
        "mean_accuracy": {paradigm: [] for paradigm in paradigms},
        "mean_error": {paradigm: [] for paradigm in paradigms},
        "median_error": {paradigm: [] for paradigm in paradigms},
        "rt_mse": {paradigm: [] for paradigm in paradigms},
        "mean_weighted_error": {paradigm: [] for paradigm in paradigms},
        "MostFreqPred_error": {paradigm: [] for paradigm in paradigms},
    }    

    # Sort and group true positions
    if rel_pos == False:
        pos_lgt_sorted = np.concatenate(u.sort_and_group(data.pos_lgt_masked, data.trial_startloc_lgt))
        pos_drk_sorted = np.concatenate(u.sort_and_group(data.pos_drk_masked, data.trial_startloc_drk))
    else:
        pos_lgt_sorted = data.relpos_lgt
        pos_drk_sorted = data.relpos_drk

    for paradigm in paradigms:
        # Set posterior
        posterior = data.POSTERIOR[paradigm]
        num_trials, num_tbins, num_pbins = posterior.shape
        # Set true and predicted positions
        if paradigm in ['lgtlgt', 'drklgt']:
            true = pos_lgt_sorted
        elif paradigm in ['drkdrk', 'lgtdrk']:
            true = pos_drk_sorted
        pred = data.DECODED_POS[paradigm]
        pred_mask = ~np.isnan(pred)

        # ----------------------------------------------------------------------
        # Confusion Matrix and Most Frequently Predicted Position Error
        # ----------------------------------------------------------------------
        MostFreqPred_error = []
        for pos in range(num_pbins):
            # Count frequency for each possible predicted position
            pred_pos_count, num_preds = r.get_pred_pos_count(true, pred, pos)
            # Compute confusion matrix
            for x, count in pred_pos_count.items():
                results['confusion_mtx'][paradigm][int(pos), int(x)] = count / num_preds
            # Compute Most Frequently Predicted Position Error
            MostFreqPred_error_pos = r.get_MostFreqPred_error(pred_pos_count, pos)
            MostFreqPred_error.append(MostFreqPred_error_pos)

        # ----------------------------------------------------------------------
        # Mean Accuracy, Mean Error, Median Error, RT-MSE, Mean Weighted Error
        # ----------------------------------------------------------------------
        trial_accuracy = []
        trial_mean_error = []
        trial_median_error = []
        trial_rt_mse = []
        trial_wt_error = []

        for trial in range(num_trials):
            accuracy = r.get_trial_accuracy(true, pred, pred_mask, trial)
            errors = r.get_trial_errors(true, pred, pred_mask, trial)
            wt_error = r.get_trial_wt_error(true, posterior, trial, num_tbins, num_pbins)

            trial_accuracy.append(accuracy)
            trial_mean_error.append(np.nanmean(errors))
            trial_median_error.append(np.nanmedian(errors))
            trial_rt_mse.append(np.sqrt(np.nanmean(np.square(errors))))
            trial_wt_error.append(wt_error)

        # ----------------------------------------------------------------------
        # Averaging results
        # ----------------------------------------------------------------------
        results["mean_accuracy"][paradigm] = np.nanmean(trial_accuracy)
        results["mean_error"][paradigm] = np.nanmean(trial_mean_error)
        results["median_error"][paradigm] = np.nanmean(trial_median_error)
        results["rt_mse"][paradigm] = np.nanmean(trial_rt_mse)
        results["mean_weighted_error"][paradigm] = np.nanmean(trial_wt_error)
        results["MostFreqPred_error"][paradigm] = np.nanmean(MostFreqPred_error)

    return results




def save_and_plot(
        data, # data object e.g. d.Data
        rel_pos: bool = False,
        saveoutput: bool = True,
        savecsv: bool = True,
        savefig: bool = True,
        save_dir: str = None,
):
    """
    """
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']
    tau = int(data.tau * 1000)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save decoding outputs: posterior, decoded positions, confusion matrix, true positions
    if saveoutput == True:
        if rel_pos == False:
            u.save_dict_to_h5(data.POSTERIOR, f"{save_dir}/{data.mouse_ID}_{data.date}_{data.area}_posterior.h5")
            u.save_dict_to_h5(data.DECODED_POS, f"{save_dir}/{data.mouse_ID}_{data.date}_{data.area}_decoded_pos.h5")
            u.save_dict_to_h5(data.results['confusion_mtx'], f"{save_dir}/{data.mouse_ID}_{data.date}_{data.area}_confusion_mtx.h5")
            u.save_dict_to_h5({'lgt': data.pos_lgt_masked, 'drk': data.pos_drk_masked}, f"{save_dir}/{data.mouse_ID}_{data.date}_{data.area}_true_pos.h5")
        else:
            u.save_dict_to_h5(data.POSTERIOR, f"{save_dir}/{data.mouse_ID}_{data.date}_{data.area}_relpos_posterior.h5")
            u.save_dict_to_h5(data.DECODED_POS, f"{save_dir}/{data.mouse_ID}_{data.date}_{data.area}_relpos_decoded_pos.h5")
            u.save_dict_to_h5(data.results['confusion_mtx'], f"{save_dir}/{data.mouse_ID}_{data.date}_{data.area}_relpos_confusion_mtx.h5")
            u.save_dict_to_h5({'lgt': data.relpos_lgt, 'drk': data.relpos_drk}, f"{save_dir}/{data.mouse_ID}_{data.date}_{data.area}_relpos_true_pos.h5")

    # Create DataFrame
    metrics = {r: [data.results[r][p] for p in paradigms] 
               for r in data.results if r != 'confusion_mtx'}
    df = pd.DataFrame(metrics, index=paradigms)
    df.index.name = 'paradigm'
    print("\nDecoding results:")
    print(df.to_markdown())

    # Export to CSV
    if savecsv == True:
        if rel_pos == False:
            filename = f"{data.mouse_ID}_{data.date}_{tau}ms_{data.area}_{data.num_units}_units_results.csv"
        else:
            filename = f"{data.mouse_ID}_{data.date}_{tau}ms_{data.area}_{data.num_units}_units_results_relpos.csv"
        df.to_csv(save_dir / filename)

    # Plot confusion matrix
    for paradigm in paradigms:
        filepath = None
        if rel_pos == False:
            if savefig == True:
                filepath = save_dir / f"{data.mouse_ID}_{data.date}_{tau}ms_{data.area}_{data.num_units}_units_confusion_mtx_{paradigm}.png"
            figs.plot_confusion_mtx(data.results['confusion_mtx'][paradigm], paradigm, savefig, filepath)
        else:
            if savefig == True:
                filepath = save_dir / f"{data.mouse_ID}_{data.date}_{tau}ms_{data.area}_{data.num_units}_units_relpos_confusion_mtx_{paradigm}.png"
            figs.plot_relative_confusion_mtx(data.results['confusion_mtx'][paradigm], paradigm, savefig, filepath)

