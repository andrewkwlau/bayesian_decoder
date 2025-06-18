import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath('../library/Npxl'))
import data_npxl as d
import utils_npxl as u

def plot_single_tuning(
        mouse: d.NpxlData,
        data: np.ndarray,
        data_type: str,
        neuron_idx: int = None,
        save: bool = False,
        filepath: str = None
):
    """
    Plot single neuron spatial tuning curves averaged across both light and dark trials.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        data (np.ndarray):
            Data matrix that is already position-binned.

        data_type (str):
            The two different output types from CASCADE.
            - 'spikeprob' = spike probability.
            - 'spikes'= discrete spikes.

        neuron_idx (int):
            The neuron to be plotted. Default random.

        save (bool):
            Whether to save the figure. Default False.
    """
    num_trials, num_pbins, num_neurons = data.shape

    if neuron_idx == None:
        neuron_idx = np.random.randint(0, num_neurons)

    data_light, data_dark = u.split_lightdark(data, mouse.darktrials)

    # --------------------------------------------------------------------------
    # Generate Data Frame
    # --------------------------------------------------------------------------
    position_all = np.tile(np.arange(num_pbins), num_trials)
    condition_all = np.array(
        ['light'] * data_light.shape[0] * num_pbins + 
        ['dark'] * data_dark.shape[0] * num_pbins
    )
    data_all = np.concatenate(
        (data_light[:,:,neuron_idx].flatten(), data_dark[:,:,neuron_idx].flatten())
    )    
    df = pd.DataFrame(
        {'position':position_all, 'condition':condition_all, 'activity':data_all}
    )

    # --------------------------------------------------------------------------
    # Plotting the figure
    # --------------------------------------------------------------------------
    # Initialise figure
    fig, (ax1) = plt.subplots(1, figsize=(6,3), sharex=True)
    fig.tight_layout(pad=2.5, h_pad=2)

    # Plot
    ax1 = sns.lineplot(data=df, x ='position', y='activity', hue='condition', palette=['orange','navy'])

    # Figure settings
    ax_settings = {
        'title':'Spatial Tuning Curve of Neuron {}'.format(neuron_idx),
        'xlabel':'position (cm)',
        'xticks':[-0.5, 9.5, 19.5, 29.5, 39.5, 49.5],
        'xticklabels':[0, 100, 200, 300, 400, 500]
    }
    if data_type == "spikeprob":        
        ax1.set(ylabel ='spike probability', **ax_settings)
    elif data_type == "spikes":
        ax1.set(ylabel = 'firing rate(Hz)', **ax_settings)

    # Labelling landmarks and reward zone
    landmark = [(10.5,12.5), (18.5,20.5), (26.5,28.5), (34.5,36.5), (43.5,45.5)]
    for coord in landmark:
        ax1.axvspan(coord[0], coord[1], alpha=0.1, color='grey')

    rewardzone = (46, 49)
    ax1.axvspan(rewardzone[0], rewardzone[1], alpha=0.1, color='green')

    # Seaborn doesn't plot NaNs, add white line to show position 0 properly
    ax1.axvline(0, linewidth=1, color="white")

    if save == True:
        plt.savefig(filepath, dpi=300)     
    plt.show()


def plot_single_heatmap(
        mouse:  d.NpxlData,
        data: np.ndarray,
        data_type: str,
        neuron_idx: int = None,
        save: bool = False,
        filepath: str = None
):
    """
    Plot single neuron spatial tuning curves heatmap for both light and dark trials.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        data (np.ndarray):
            Data matrix that is already position-binned.

        data_type (str):
            The two different output types from CASCADE.
            - 'spikeprob' = spike probability.
            - 'spikes'= discrete spikes.

        neuron_idx (int):
            The neuron to be plotted. Default random.

        save (bool):
            Whether to save the figure. Default False.
    """
    # Split light and dark trials
    data_light, data_dark = u.split_lightdark(data, mouse.darktrials)
    
    # Sort data by trial start position
    sorted_data_light = u.sort_trialstart(
        data_light[:,:,neuron_idx], mouse.position_mtx, mouse.darktrials, 'light'
    )
    sorted_data_dark = u.sort_trialstart(
        data_dark[:,:,neuron_idx], mouse.position_mtx, mouse.darktrials, 'dark'
    )

    # --------------------------------------------------------------------------
    # Plotting the figure
    # --------------------------------------------------------------------------
    # Initialise figure
    fig, (ax1, ax2) = plt.subplots(1,2, sharex=True)
    fig.suptitle('Spike Activity of Neuron {} Across Trials'.format(neuron_idx), y=0.94)
    fig.tight_layout(pad=2, w_pad=2)
       
    # Plot    
    if data_type == "spikeprob":
        im1 = ax1.imshow(sorted_data_light, cmap='turbo')
        im2 = ax2.imshow(sorted_data_dark, cmap="turbo")        
    elif data_type == "spikes":
        im1 = ax1.imshow(sorted_data_light, cmap='turbo')
        im2 = ax2.imshow(sorted_data_dark, cmap="turbo")

    # Figure Settings
    ax_settings = {
        'xlabel':'position (cm)',
        'ylabel':'trials',
        'xticks':[-0.5, 9.5, 19.5, 29.5, 39.5, 49.5],
        'xticklabels':[0, 100, 200, 300, 400, 500],
        'aspect':'auto',
        'facecolor':'black'
    }     
    ax1.set(title='light trials', **ax_settings)
    ax2.set(title='dark trials', **ax_settings)

    # Labelling landmarks and rewardzone
    landmark = [10.5, 12.5, 18.5, 20.5, 26.5, 28.5, 34.5, 36.5, 43.5, 45.5]
    for x in landmark:
        ax1.axvline(x, linestyle="dashed", linewidth=1, color="white")
        ax2.axvline(x, linestyle="dashed", linewidth=1, color="white")

    rewardzone = [46, 49]
    for x in rewardzone:
        ax1.axvline(x, linestyle="dashed", linewidth=1, color="magenta")
        ax2.axvline(x, linestyle="dashed", linewidth=1, color="magenta")
    ax1.axvspan(rewardzone[0], rewardzone[1], alpha=0.1, color='magenta')
    ax2.axvspan(rewardzone[0], rewardzone[1], alpha=0.1, color='magenta')    

    # Add colour bar
    cbar = plt.colorbar(im1, ax=ax2)
    if data_type == "spikeprob":
        cbar.set_label("spike probability")
    elif data_type == "spikes":
        cbar.set_label("firing rate (Hz)")

    if save == True:
        plt.savefig(filepath, dpi=300)   
    plt.show()


def plot_confusion_mtx(
        mouse:  d.NpxlData,
        confusion_mtx: np.ndarray,
        paradigm: str,
        save: bool = False,
        filepath: str = None
):
    """
    Plot confusion matrix.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        confusion_mtx (np.ndarray):
            The confusion_mtx to be plotted.

        paradigm (str):
            The train/test paradigm of the confusion_mtx:
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

        save (bool):
            Whether to save the figure. Default False.
    """
    # Initialise figure
    fig, (ax1) = plt.subplots(1, figsize=(6,6))
    fig.tight_layout(pad=1)

    # Plot
    im1 = ax1.imshow(confusion_mtx, cmap="turbo", vmax=1)

    # Figure Settings
    ax_settings = {
        'xlabel':'decoded position (cm)',
        'ylabel':'true position (cm)',
        'xticks':[-0.5, 9.5, 19.5, 29.5, 39.5, 45.5],
        'xticklabels':[0, 100, 200, 300, 400, 460],
        'yticks':[-0.5, 9.5, 19.5, 29.5, 39.5, 45.5],
        'yticklabels':[0, 100, 200, 300, 400, 460]
    }   
    if paradigm == 'lgtlgt':
        ax1.set(title = 'Trained in Light, Test in Light', **ax_settings)
    elif paradigm == 'drkdrk':
        ax1.set(title = 'Trained in Dark, Test in Dark', **ax_settings)
    elif paradigm == 'lgtdrk':
        ax1.set(title = 'Trained in Light, Test in Dark', **ax_settings)
    elif paradigm == 'drklgt':
        ax1.set(title = 'Trained in Dark, Test in Light', **ax_settings)

    # Labelling landmarks
    landmark = [11, 13, 19, 21, 27, 29, 35, 37, 44, 46]
    for x in landmark:
        ax1.axvline(x-0.5, linestyle="dashed", linewidth=1, color="white", alpha=0.5) 
        ax1.axhline(x-0.5, linestyle="dashed", linewidth=1, color="white", alpha=0.5)

    # Add Diagonal
    ax1.plot([1, 0], [0, 1], transform=ax1.transAxes, linewidth=1)

    # Add colour bar
    cbar = fig.colorbar(im1, ax=ax1, shrink=0.75)
    cbar.set_label("% of decoded position for each true position")

    if save == True:
        plt.savefig(filepath, dpi=300)
    plt.show()


def plot_accuracy(
        mouse:  d.NpxlData,
        decoder_results: dict,
        chance_results: dict,
        num_reps: int,
        save: bool = False,
        filepath: str = None
):
    """
    """
    # --------------------------------------------------------------------------
    # Generate Data Frame
    # --------------------------------------------------------------------------
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']

    accuracy_decoder = decoder_results['mean_accuracy']
    accuracy_chance = chance_results['accuracy_allreps']
    accuracy_combined = {paradigm:[] for paradigm in paradigms}
    for paradigm in paradigms:
        accuracy_combined[paradigm] = accuracy_decoder[paradigm] + accuracy_chance[paradigm]

    accuracy_all = []
    paradigm_all = []
    source_all = []

    for paradigm in paradigms:
        accuracy_all += accuracy_combined[paradigm]
        paradigm_all += ([paradigm] * (num_reps + 1))
        source_all += ['decoder'] + ['chance']*num_reps

    df = pd.DataFrame({'accuracy':accuracy_all, 'paradigm':paradigm_all, 'source':source_all})
    df['accuracy'] = df['accuracy']*100

    # --------------------------------------------------------------------------
    # Plotting the figure
    # --------------------------------------------------------------------------
    # Initialise figure
    fig, (ax1) = plt.subplots(1, figsize=(6,4))
    fig.tight_layout(pad=2.5, h_pad=2)

    # Plot
    ax1 = sns.barplot(data=df, x='paradigm', y='accuracy', hue='source', palette='BuGn', errorbar='sd')

    # Figure Settlings
    ax_settings = {
        'title':'Decoder accuracy compared to chance level estimate',
        'xlabel':'Train/Test Paradigms',
        'ylabel':'Accuracy (%)'
    }
    ax1.set(**ax_settings)

    # Legends
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=labels)

    if save == True:
        plt.savefig(filepath, dpi=300)

    plt.show()


def plot_errors(
        mouse:  d.NpxlData,
        decoder_results: dict,
        chance_results: dict,
        num_reps: int,
        save: bool = False,
        filepath: str = None
):
    """
    """
    # --------------------------------------------------------------------------
    # Generate Data Frame
    # --------------------------------------------------------------------------
    paradigms = ['lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt']

    error_decoder = decoder_results['mean_error']
    errors_chance = chance_results['errors_allreps']
    errors_combined = {paradigm:[] for paradigm in paradigms}
    for paradigm in paradigms:
        errors_combined[paradigm] = error_decoder[paradigm] + errors_chance[paradigm]

    errors_all = []
    paradigm_all = []
    source_all = []

    for paradigm in paradigms:
        errors_all += errors_combined[paradigm]
        paradigm_all += ([paradigm] * (num_reps + 1))
        source_all += ['decoder'] + ['chance']*num_reps

    df = pd.DataFrame({'errors':errors_all, 'paradigm':paradigm_all, 'source':source_all})

    # --------------------------------------------------------------------------
    # Plotting the figure
    # --------------------------------------------------------------------------
    # Initialise figure
    fig, (ax1) = plt.subplots(1, figsize=(6,4))
    fig.tight_layout(pad=2.5, h_pad=2)

    # Plot
    ax1 = sns.barplot(data=df, x='paradigm', y='errors', hue='source', palette='OrRd', errorbar='se')

    # Figure Settings
    ax_settings = {
        'title':'Decoder absolute errors compared to chance level estimate',
        'xlabel':'Train/Test Paradigms',
        'ylabel':'Absolute errors (10cm Position Bins)',
        'ylim':(0,15)
    }
    ax1.set(**ax_settings)

    # Legends
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=labels)

    if save == True:
        plt.savefig(filepath, dpi=300)

    plt.show()


def plot_pca_3d(
        mouse,
        num_chunks: int,
        trial_avg: bool = False
):
    """
    """
    chunk_titles = np.sort([f'Chunk {i}' for i in range(num_chunks)] * 2)

    # Create a subplot figure with 10 rows, 2 columns (one column for each condition)
    fig = make_subplots(rows=10, cols=2, 
                        shared_xaxes=True,
                        subplot_titles=chunk_titles,
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]] * 10,
                        vertical_spacing=0.01)
    
    # Loop through chunks
    for chunk in range(num_chunks):
        data_lgt = mouse.fr_lgt_reconstructed
        data_drk = mouse.fr_drk_reconstructed

        # Loop through each trial and add a scatter plot for the points
        for trial in range(data_lgt[chunk].shape[0]):
            if trial_avg == True:
                x = np.nanmean(data_lgt[chunk], axis=0)[:, 0]
                y = np.nanmean(data_lgt[chunk], axis=0)[:, 1]
                z = np.nanmean(data_lgt[chunk], axis=0)[:, 2]
            else:
                x = data_lgt[chunk][trial, :, 0]
                y = data_lgt[chunk][trial, :, 1]
                z = data_lgt[chunk][trial, :, 2]

            # Create a gradient color based on the index
            colors = np.linspace(0, 1, len(x))  # Normalized from 0 to 1

            # Add trace to the correct subplot (row=chunk+1 since indexing starts from 1)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                marker=dict(size=2, color=colors, colorscale='Viridis'),
                line=dict(color=colors, colorscale='Viridis', width=3),
                mode='lines+markers'
            ), row=chunk+1, col=1)

        # Loop through each trial and add a scatter plot for the points
        for trial in range(data_drk[chunk].shape[0]):
            if trial_avg == True:
                x = np.nanmean(data_drk[chunk], axis=0)[:, 0]
                y = np.nanmean(data_drk[chunk], axis=0)[:, 1]
                z = np.nanmean(data_drk[chunk], axis=0)[:, 2]
            else:
                x = data_drk[chunk][trial, :, 0]
                y = data_drk[chunk][trial, :, 1]
                z = data_drk[chunk][trial, :, 2]

            # Create a gradient color based on the index
            colors = np.linspace(0, 1, len(x))  # Normalized from 0 to 1

            # Add trace to the correct subplot (row=chunk+1 since indexing starts from 1)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                marker=dict(size=2, color=colors, colorscale='Viridis'),
                line=dict(color=colors, colorscale='Viridis', width=3),
                mode='lines+markers'
            ), row=chunk+1, col=2)
        
    # Update layout
    fig.update_layout(
        height=4000,  # Adjust height to fit all subplots
        width=1000,
        title=f'Plot of Light and Dark PCA Components for {mouse.mouse_ID}',
        showlegend=False
    )

    # Show the figure
    fig.show()
