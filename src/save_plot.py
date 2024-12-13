# save_plot.py
# author: Ximin Xu
# date: 2024-12-13

import os

def save_plot(figure, directory, filename="plot.png"):
    """
    Saves a given plot to a specified directory.

    Parameters:
        figure (matplotlib.figure.Figure): The plot figure to save.
        directory (str): Path to the directory where the plot will be saved.
        filename (str): Name of the plot file. Default is 'plot.png'.

    Returns:
        None
    """
    if not os.path.isdir(directory):
        raise ValueError('The directory provided does not exist.')
    try:
        figure.savefig(os.path.join(directory, filename))
    except Exception as e:
        raise ValueError(f"Error saving the plot: {e}")