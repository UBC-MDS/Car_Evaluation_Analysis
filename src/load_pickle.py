# load_pickle.py
# author: Ximin Xu
# date: 2024-12-13

import pickle
import os


def load_pickle(file_path):
    """
    Loads an object from a pickle file.

    Parameters:
        file_path (str): Path to the pickle file.

    Returns:
        object: The loaded pickle.

    Raises:
        ValueError: If the input pickle is not valid.
    """
    if not os.path.isfile(file_path):
        raise ValueError('The file provided does not exist.')

    if not file_path.endswith('.pkl') and not file_path.endswith('.pickle'):
        raise ValueError('The file provided is not a pickle file.')
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise ValueError(f"Error loading the pickle file: {e}")