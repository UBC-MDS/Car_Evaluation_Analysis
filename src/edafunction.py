import os
import altair as alt
import pandas as pd


def save_altair_plot(plot, directory, filename="eda.png"):
    """
    Saves an Altair plot to a specified directory.

    Parameters:
        plot (alt.Chart): The Altair plot to save.
        directory (str): Path to the directory where the plot will be saved.
        filename (str): Name of the plot file. Default is 'eda.png'.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If there is an error during the save process.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    try:
        plot.save(os.path.join(directory, filename), scale_factor=2.0)
    except Exception as e:
        raise ValueError(f"Error saving the plot: {e}")


def create_target_distribution_plot(data):
    """
    Creates an Altair bar chart for the target variable distribution.

    Parameters:
        data (pd.DataFrame): The raw data containing the 'class' column.

    Returns:
        alt.Chart: The Altair chart object.
    """
    if 'class' not in data.columns:
        raise ValueError("The dataset must include a 'class' column as the target variable.")

    return alt.Chart(data).mark_bar().encode(
        x=alt.X('class', title='Target Class'),
        y=alt.Y('count()', title='Count'),
        color=alt.Color('class', legend=None)
    ).properties(
        title="Target Variable Distribution",
        width=400,
        height=300
    )


def create_feature_distributions_plot(data, features):
    """
    Creates an Altair grid of bar charts for feature distributions by class.

    Parameters:
        data (pd.DataFrame): The processed data containing the features and 'class' column.
        features (list): A list of feature column names to plot.

    Returns:
        alt.Chart: The Altair chart object.
    """
    if 'class' not in data.columns:
        raise ValueError("The dataset must include a 'class' column as the target variable.")

    return alt.Chart(data).mark_bar().encode(
        x=alt.X(alt.repeat('row')),
        y='count()',
        color=alt.Color('class', title='Class'),
        column='class'
    ).properties(
        height=100
    ).repeat(
        row=features
    )


def run_eda_charts(figures_path, raw_data_path, processed_data_path):
    """
    Main function to create and save EDA charts.

    Parameters:
        figures_path (str): The directory path where all the plots will be saved.
        raw_data_path (str): The path to the raw dataset CSV file.
        processed_data_path (str): The path to the processed dataset CSV file.

    Returns:
        None
    """
    if not os.path.isdir(figures_path):
        raise FileNotFoundError(f"The directory '{figures_path}' does not exist.")

    # Load raw and processed data
    raw_data = pd.read_csv(raw_data_path)
    processed_data = pd.read_csv(processed_data_path)

    # Create and save target distribution plot
    target_plot = create_target_distribution_plot(raw_data)
    save_altair_plot(target_plot, figures_path, "target_distribution_raw.png")

    # Create and save feature distributions plot
    features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    feature_plot = create_feature_distributions_plot(processed_data, features)
    save_altair_plot(feature_plot, figures_path, "feature_counts_by_class.png")
