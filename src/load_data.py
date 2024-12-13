# load_data.py
# author: Ximin Xu
# date: 2024-12-13

import pandas as pd
import os

def load_data(file_path):
    """
    Loads data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
        
    Raises:
        ValueError: If the input data is not valid.
    """
    if not os.path.isfile(file_path):
        raise ValueError('The file provided does not exist.')
    
    if not file_path.endswith('.csv'):
        raise ValueError('The file provided is not a CSV file.')
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")
    
    return data
