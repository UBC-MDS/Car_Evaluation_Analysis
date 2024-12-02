# download_data.py
# author: Nicholas Varabioff
# data: 2024-02-12

# import raw data
# data located at https://archive.ics.uci.edu/dataset/19/car+evaluation
# requirements: `pip install ucimlrepo`

import click
import os
from ucimlrepo import fetch_ucirepo

@click.command()
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")


def main(data_to):
    '''
    This script downloads the raw data from the UCI source.
    It saves the data files to the local repository.
    '''
    # fetch dataset
    car_evaluation = fetch_ucirepo(id=19)

    # data (as pandas dataframes)
    X = car_evaluation.data.features
    y = car_evaluation.data.targets

    # save data
    X.to_csv(os.path.join(data_to, 'car_features_raw.csv'))
    y.to_csv(os.path.join(data_to, 'car_targets_raw.csv'))

    # print metadata
    # print(car_evaluation.metadata)

    # print variable information
    # print(car_evaluation.variables)


if __name__ == '__main__':
    main()