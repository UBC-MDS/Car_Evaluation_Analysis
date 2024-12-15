import os
import pandas as pd


def save_table(dataframe, directory_path, table_name):
    """
    Saves a DataFrame as a CSV file named table_name in the specified directory.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        directory_path (str): The path to the directory where the CSV file will be saved.
        table_name (str): The name of the saved csv

    Returns:
        None

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the input/output is not a valid DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The provided data is not a valid pandas DataFrame.")

    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")

    if not table_name.endswith('.csv'):
        raise ValueError("The table_name must end with '.csv'.")

    csv_path = os.path.join(directory_path, table_name)
    dataframe.to_csv(csv_path, index=False)
