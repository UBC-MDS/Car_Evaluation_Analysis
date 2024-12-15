# eda.py
# author: Zuer Zhong
# date: 2024-12-03
import click
import os
import sys
import altair as alt
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_data import load_data
from src.edafunction import save_altair_plot


@click.command()
@click.option('--raw-data', type=str, required=True, help="Path to raw data (before preprocessing)")
@click.option('--processed-training-data', type=str, required=True, help="Path to processed training data")
@click.option('--plot-to', type=str, required=True, help="Path to directory where the plot will be written to")
def main(raw_data, processed_training_data, plot_to):
    '''Plots the count of each feature in the processed training data
        by class and displays them as a grid of plots. Also saves the plot.'''

    car_data_raw = load_data(raw_data)
    raw_target_distribution = alt.Chart(car_data_raw).mark_bar().encode(
        x=alt.X('class', title='Target Class (Raw Data)'),
        y=alt.Y('count()', title='Count'),
        color=alt.Color('class', legend=None)
    ).properties(
        title='Target Variable Distribution (Raw Data)',
        width=300,
        height=200
    )
    save_altair_plot(raw_target_distribution, plot_to, filename="target_distribution_raw.png")

    car_train = load_data(processed_training_data)
    plot = alt.Chart(car_train).mark_bar().encode(
        x=alt.X(alt.repeat('row')),
        y='count()',
        color=alt.Color('class', title='Class'),
        column='class'
    ).properties(
        height=100
    ).repeat(
        row=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    )
    save_altair_plot(plot, plot_to, filename="feature_counts_by_class.png")


if __name__ == '__main__':
    main()
