# eda.py
# author: Zuer Zhong
# date: 2024-12-03
import click
import os
import altair as alt
import pandas as pd


@click.command()
@click.option('--raw-data', type=str, required=True, help="Path to raw data (before preprocessing)")
@click.option('--processed-training-data', type=str, required=True, help="Path to processed training data")
@click.option('--plot-to', type=str, required=True, help="Path to directory where the plot will be written to")
def main(raw_data, processed_training_data, plot_to):
    '''Plots the count of each feature in the processed training data
        by class and displays them as a grid of plots. Also saves the plot.'''

    car_data_raw = pd.read_csv(raw_data)
    raw_target_distribution = alt.Chart(car_data_raw).mark_bar().encode(
        x=alt.X('class', title='Target Class (Raw Data)'),
        y=alt.Y('count()', title='Count'),
        color=alt.Color('class', legend=None)
    ).properties(
        title='Target Variable Distribution (Raw Data)',
        width=300,
        height=200
    )
    raw_target_distribution.save(os.path.join(plot_to, "target_distribution_raw.png"), scale_factor=2.0)

    car_train = pd.read_csv(processed_training_data)
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

    plot.save(os.path.join(plot_to, "feature_counts_by_class.png"), scale_factor=2.0)


if __name__ == '__main__':
    main()
