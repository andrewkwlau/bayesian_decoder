import sys
import os
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath('../library'))
import data as d
import utils as u


def generate_confusion_mtx(
        mouse,
        decoded_pos: np.ndarray, 
        paradigm: str, 
        num_pbins: int
) -> np.ndarray:
    """
    Generate confusion matrix. Y-axis: true positions, X-axis: decoded positions.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        decoded_pos (np.ndarray):
            Decoded position matrix.

        paradigm (str):
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'.

        num_pbins (float):
            Number of decoded position bins excluding reward zone.

    Returns:
        confusion_mtx (np.ndarray):
            Y-axis: true positions, X-axis: decoded positions.
            Each value represents the percentage of decoded position x over all 
            decoded position for true position y.

    """
    confusion_mtx = np.zeros((num_pbins,num_pbins))

    # Specify true position and predicted position
    if paradigm == 'lgtlgt' or paradigm == 'drklgt':
        true = mouse.pos_lgt_masked
    elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':
        true = mouse.pos_drk_masked
    pred = decoded_pos
        
    # For each true position bin y / for each row in the y-axis
    for y in range(num_pbins):        
        # Find all the time bins where true position is y
        true_occurrence_idx = list(zip(*np.where(true == y)))
        # Find corresponding predicted positions in those time bins
        predictions = [pred[i] for i in true_occurrence_idx]
        # Count occurence of each predicted position, dict(position: no. of occurence)
        prediction_pos_count = Counter(predictions)
        
        # For each predicted position bin x / for each col in the x-axis
        for x, count in prediction_pos_count.items(): 
            if np.isnan(x):
                continue                
            else:
                # Compute % for count of each predicted position / total number of predictions
                confusion_mtx[int(y),int(x)] = (count / len(predictions))
    
    print("confusion_mtx max: ", np.nanmax(confusion_mtx))
    print("confusion_mtx min: ", np.nanmin(confusion_mtx))

    return confusion_mtx


def generate_confusion_mtx_allchunks(
        mouse: d.CaImgData | d.NpxlData,
        decoded_pos_allchunks: list,
        num_pbins: int,
        num_chunks: int,
        discrete: bool = True
) -> dict:
    """
    Generate one confusion matrix for all chunks. Y-axis: true positions, X-axis: decoded positions.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        decoded_pos_allchunks (np.ndarray):
            Decoded positions of all chunks and all paradigms

        num_pbins (float):
            Number of decoded position bins excluding reward zone.

        num_chunks (int):
            Number of chunks to be divided.

    Returns:
        confusion_mtx (dict):
            a dict containing confusion matrices for each train/test paradigm.
            Y-axis: true positions, X-axis: decoded positions.
            Each value represents the percentage of decoded position x over all 
            decoded position for true position y.
    
    """   
    # Chunk position matrix
    pos_light_chunks = u.sort_and_chunk(mouse, mouse.pos_lgt_masked, 'lgt', discrete, num_chunks)
    pos_dark_chunks = u.sort_and_chunk(mouse, mouse.pos_drk_masked, 'drk', discrete, num_chunks)

    # Intialise confusion matrix for each train/test paradigm
    confusion_mtx_allchunks = {
    'lgtlgt': np.zeros((num_pbins, num_pbins)),
    'drkdrk': np.zeros((num_pbins, num_pbins)),
    'lgtdrk': np.zeros((num_pbins, num_pbins)),
    'drklgt': np.zeros((num_pbins, num_pbins))
    }

    # Intialise list of decoder predictions for all true position bins in the y-axis
    # for each train/test paradigm
    predictions_for_y = {
    'lgtlgt': {y: [] for y in range(num_pbins)},
    'drkdrk': {y: [] for y in range(num_pbins)},
    'lgtdrk': {y: [] for y in range(num_pbins)},
    'drklgt': {y: [] for y in range(num_pbins)}
    }

    num_chunks = len(pos_light_chunks)
    # Loop through each chunk and train/test paradigm
    for i in range(num_chunks):
        for paradigm in decoded_pos_allchunks[i]:
            # Set true position
            if paradigm == 'lgtlgt' or paradigm == 'drklgt':
                true = pos_light_chunks[i]
            elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':
                true = pos_dark_chunks[i]
            # Set predicted position
            pred = decoded_pos_allchunks[i][paradigm]

            # For each true position bin y / for each row in the y-axis
            for y in range(num_pbins):        
                # Find all the time bins where true position is y
                true_occurrence_idx = list(zip(*np.where(true == y)))
                # Find corresponding predicted positions in those time bins
                predictions = [pred[i] for i in true_occurrence_idx]
                # Store the predicted positions
                predictions_for_y[paradigm][y].extend(predictions)


    for paradigm in confusion_mtx_allchunks:
        # For each true position bin y / for each row in the y-axis
        for y in range(num_pbins):
            # Count number of predictions for each predicted position, dict{position: no. of occurence}
            prediction_pos_count = Counter(predictions_for_y[paradigm][y])
            total_num_predictions = len(predictions_for_y[paradigm][y])

            # For each predicted position bin x / for each col in the x-axis
            for x, count in prediction_pos_count.items():
                    if np.isnan(x):
                        continue
                    else:
                        # Compute % for count of each predicted position / total number of predictions
                        confusion_mtx_allchunks[paradigm][int(y), int(x)] = count / total_num_predictions

    return confusion_mtx_allchunks


def generate_confusion_mtx_perchunk(
        mouse: d.CaImgData | d.NpxlData,
        decoded_pos_chunk: np.ndarray,
        paradigm: str,
        num_pbins: int,
        num_chunks: int,
        chunk_idx: int,
        discrete: bool = True
):
    """
    Generate confusion matrix for each chunk. Y-axis: true positions, X-axis: decoded positions.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        decoded_pos_chunk (np.ndarray):
            Decoded position of a chunk.

        paradigm (str):
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'.

        num_pbins (int):
            Number of decoded position bins excluding reward zone.

        num_chunks (int):
            Number of chunks to be divided.

        chunk_idx (int):
            The chunk to generate confusion matrix for.

    Returns:
        confusion_mtx_chunk (np.ndarray):
            Y-axis: true positions, X-axis: decoded positions.
            Each value represents the percentage of decoded position x over all 
            decoded position for true position y.
    """
    confusion_mtx_chunk = np.zeros((num_pbins,num_pbins))

    # Chunk position matrix
    pos_light_chunks = u.sort_and_chunk(mouse, mouse.pos_lgt_masked, 'lgt', discrete, num_chunks)
    pos_dark_chunks = u.sort_and_chunk(mouse, mouse.pos_drk_masked, 'drk', discrete, num_chunks)

    # Specify true position and predicted position
    if paradigm == 'lgtlgt' or paradigm == 'drklgt':
        true = pos_light_chunks[chunk_idx]
    elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':
        true = pos_dark_chunks[chunk_idx]
    pred = decoded_pos_chunk
        
    # For each true position bin y / for each row in the y-axis
    for y in range(num_pbins):        
        # Find all the time bins where true position is y
        true_occurrence_idx = list(zip(*np.where(true == y)))
        # Find corresponding predicted positions in those time bins
        predictions = [pred[i] for i in true_occurrence_idx]
        # Count occurence of each predicted position, dict(position: no. of occurence)
        prediction_pos_count = Counter(predictions)
        
        # For each predicted position bin x / for each col in the x-axis
        for x, count in prediction_pos_count.items(): 
            if np.isnan(x) == True:
                continue                
            else:
                # Compute % for count of each predicted position / total number of predictions
                confusion_mtx_chunk[int(y),int(x)] = (count / len(predictions))
    
    print("confusion_mtx max: ", np.nanmax(confusion_mtx_chunk))
    print("confusion_mtx min: ", np.nanmin(confusion_mtx_chunk))

    return confusion_mtx_chunk


def compute_accuracy(
        mouse,
        decoded_pos: np.ndarray,
        paradigm: str
) -> float:
    """
    Compute accuracy rate of decoded position over non-NaN true position.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        decoded_pos (np.ndarray):
            Decoded position matrix.

        paradigm (str):
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'.

    Returns:
        accuracy_rate (float):
            Accuracy rate across all predictions.
    """   
    # Check number of NaNs and non-NaNs in decoded positions
    num_nans = np.sum(np.isnan(decoded_pos))
    num_non_nans = np.sum(~np.isnan(decoded_pos)) 
    
    # Specify true positions
    if paradigm == 'lgtlgt' or paradigm == 'drklgt':
        true = mouse.pos_lgt_masked
    elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':
        true = mouse.pos_drk_masked

    # Create a mask for time bins where there are decoder predictions
    predictions = ~np.isnan(decoded_pos)
    
    # Check for time bins with valid predictions, is decoded position matching true 
    accuracy_mtx = decoded_pos[predictions] == true[predictions]       
    accuracy_rate = np.sum(accuracy_mtx) / (len(accuracy_mtx))

    print("number of NaNs in decoded position:", num_nans)
    print("number of non-NaNs in decoded position:", num_non_nans)
    print("total decoder predictions:", accuracy_mtx.shape[0])  
    print("accuracy rate excluding NaN decoded positions:", accuracy_rate)
    
    return accuracy_rate


def compute_accuracy_chunk(
        mouse,
        decoded_pos_chunk: np.ndarray,
        paradigm: str,
        num_chunks: int,
        chunk_idx: int,
        discrete: bool = True
) -> float:
    """
    Compute accuracy rate for each chunk of decoded position over non-NaN true position.

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        decoded_pos_chunk (np.ndarray):
            Decoded position of a chunk.

        paradigm (str):
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

        num_chunks (int):
            Number of chunks to be divided.

        chunk_idx (int):
            The chunk to generate confusion matrix for.        

    Returns:
        accuracy_rate_chunk (float):
    """
    # Check number of NaNs and non-NaNs in decoded positions
    num_nans = np.sum(np.isnan(decoded_pos_chunk))
    num_non_nans = np.sum(~np.isnan(decoded_pos_chunk))

    # Chunk position matrix
    pos_light_chunks = u.sort_and_chunk(mouse, mouse.pos_lgt_masked, 'lgt', discrete, num_chunks)
    pos_dark_chunks = u.sort_and_chunk(mouse, mouse.pos_drk_masked, 'drk', discrete, num_chunks)

    # Specify true position
    if paradigm == 'lgtlgt' or paradigm == 'drklgt':  
        true = pos_light_chunks[chunk_idx]
    elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':    
        true = pos_dark_chunks[chunk_idx]

    # Create a mask for time bins where there are decoder predictions
    predictions = ~np.isnan(decoded_pos_chunk)

    # Check for time bins with valid predictions, is decoded position matching true             
    accuracy_mtx = decoded_pos_chunk[predictions] == true[predictions]             
    accuracy_rate_chunk = np.sum(accuracy_mtx) / (len(accuracy_mtx))

    print("number of NaNs in decoded position:", num_nans)
    print("number of non-NaNs in decoded position:", num_non_nans)
    print("total decoder predictions:", accuracy_mtx.shape[0])  
    print("accuracy rate excluding NaN decoded positions:", accuracy_rate_chunk)
    
    return accuracy_rate_chunk


def compute_errors(
        mouse,
        decoded_pos: np.ndarray, 
        paradigm: str
) -> tuple:
    """
    Compute errors between decoded position and true position:
    - absolute errors
    - mean squared error
    - root mean squared error

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        decoded_pos (np.ndarray):
            Decoded position matrix.

        paradigm (str):
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

    Returns:
        tuple: a tuple containing:
            - errors (np.ndarray)
            - mse (float)
            - rt_mse (float)
    """
    if paradigm == 'lgtlgt' or paradigm == 'drklgt':
        true = mouse.pos_lgt_masked
    elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':
        true = mouse.pos_drk_masked

    # Create a mask for time bins where there are decoder predictions
    predictions = ~np.isnan(decoded_pos)

    # Absolute errors between decoded position and true positions, taking into
    # consideration only the time bins where decoded position is non-NaN
    errors = abs(np.subtract(decoded_pos[predictions], true[predictions]))

    min_error = np.nanmin(errors)
    max_error = np.nanmax(errors)
    mean_error = np.nanmean(errors)
    
    sqr_error = np.square(errors)    
    mse = np.nansum(sqr_error) / len(sqr_error)
    rt_mse = np.sqrt(mse)

    print(errors.shape)
    print("min error:", min_error, "position bins (10cm)")
    print("max error:", max_error, "position bins (10cm)")
    print("mean error:", mean_error, "position bins (10cm)")
    print("mean squared error:", mse)
    print("root mean suqred error:", rt_mse, "position bins (10cm)")
        
    return errors, mse, rt_mse


def compute_errors_chunk(
        mouse,
        decoded_pos_chunk: np.ndarray, 
        paradigm: str,
        num_chunks: int,
        chunk_idx: int,
        discrete: bool = True
) -> tuple:
    """
    Compute errors between decoded position and true position:
    - absolute errors
    - mean squared error
    - root mean squared error

    Args:
        mouse (MouseData):
            An instance of class MouseData. Identify which mouse's data to use.

        decoded_pos_chunk (np.ndarray):
            Decoded position of a chunk.

        paradigm (str):
            'lgtlgt', 'drkdrk', 'lgtdrk', 'drklgt'

        num_chunks (int):
            Number of chunks to be divided.

        chunk_idx (int):
            The chunk to generate confusion matrix for. 

    Returns:
        tuple: a tuple containing:
            - errors_chunk (np.ndarray)
            - mse_chunk (float)
            - rt_mse_chunk (float)
    """
    # Chunk position matrix
    pos_light_chunks = u.sort_and_chunk(mouse, mouse.pos_lgt_masked, 'lgt', discrete, num_chunks)
    pos_dark_chunks = u.sort_and_chunk(mouse, mouse.pos_drk_masked, 'drk', discrete, num_chunks)
    
    # Specify true position
    if paradigm == 'lgtlgt' or paradigm == 'drklgt':  
        true = pos_light_chunks[chunk_idx]
    elif paradigm == 'drkdrk' or paradigm == 'lgtdrk':    
        true = pos_dark_chunks[chunk_idx]

    # Create a mask for time bins where there are decoder predictions
    predictions = ~np.isnan(decoded_pos_chunk)

    # Absolute errors between decoded position and true positions, taking into
    # consideration only the time bins where decoded position is non-NaN
    errors_chunk = abs(np.subtract(decoded_pos_chunk[predictions], true[predictions]))

    min_error = np.nanmin(errors_chunk)
    max_error = np.nanmax(errors_chunk)
    mean_error = np.nanmean(errors_chunk)
    
    sqr_error = np.square(errors_chunk)    
    mse_chunk = np.nansum(sqr_error) / len(sqr_error)
    rt_mse_chunk = np.sqrt(mse_chunk)

    print(errors_chunk.shape)
    print("min error:", min_error, "position bins (10cm)")
    print("max error:", max_error, "position bins (10cm)")
    print("mean error:", mean_error, "position bins (10cm)")
    print("mean squared error:", mse_chunk)
    print("root mean suqred error:", rt_mse_chunk, "position bins (10cm)")
        
    return errors_chunk, mse_chunk, rt_mse_chunk