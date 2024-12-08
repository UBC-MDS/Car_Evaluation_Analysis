# eda.py
# author: Zuer Zhong
# date: 2024-12-03

# usage: python scripts/eda.py \
#     --processed-training-data data/processed/car_train.csv \
#     --plot-to results/figures \

import click
import os
import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--processed-training-data', type=str, required=True, help="Path to processed training data")
@click.option('--plot-to', type=str, required=True, help="Path to directory where the plot will be written to")
def main(processed_training_data, plot_to):
    '''Plots the count of each feature in the processed training data
        by class and displays them as a grid of plots. Also saves the plot.'''

    car_train = pd.read_csv(processed_training_data)

    # Ensure plot_to directory exists
    os.makedirs(plot_to, exist_ok=True)

    # exploratory data analysis - visualize predictor distributions across classes
    plot = alt.Chart(car_train).mark_bar().encode(
        x=alt.X(alt.repeat('row')),
        y='count()',
        color=alt.Color('class'),
        column='class'
    ).properties(
        height=100
    ).repeat(
        row=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    )

    plot.save(os.path.join(plot_to, "feature_counts_by_class.png"), scale_factor=2.0)

if __name__ == '__main__':
    main()