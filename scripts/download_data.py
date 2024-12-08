# download_data.py
# author: Nicholas Varabioff
# date: 2024-12-02

# import raw data
# data located at https://archive.ics.uci.edu/dataset/19/car+evaluation
# requirements: `pip install ucimlrepo`

# usage: python scripts/download_data.py \
#     --data-to data/raw

import click
import os
import pandas as pd
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
    car_data_raw = car_evaluation.data.original
    # print(car_evaluation.data.original)

    # save data
    car_data_raw.to_csv(os.path.join(data_to, 'car_data_raw.csv'))

    # print metadata
    # print(car_evaluation.metadata)

    # print variable information
    # print(car_evaluation.variables)


if __name__ == '__main__':
    main()