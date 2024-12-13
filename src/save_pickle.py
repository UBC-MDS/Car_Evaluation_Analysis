# save_pickle.py
# author: Ximin Xu
# date: 2024-12-13

import pickle
import os

def save_pickle(obj, directory, filename="object.pickle"):
    """
    Saves an object to a pickle file.

    Parameters:
        pickle (object): The pickle to save.
        directory (str): Path to the directory where the pickle file will be saved.
        filename (str): Name of the pickle file. Default is 'object.pickle'.

    Returns:
        None
    """
    if not os.path.isdir(directory):
        raise ValueError('The directory provided does not exist.')
    try:
        with open(os.path.join(directory, filename), "wb") as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise ValueError(f"Error saving the object: {e}")