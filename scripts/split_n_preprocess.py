# split_n_preprocess.py
# author: Nicholas Varabioff
# date: 2024-12-02

import click
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_validation import run_data_validation
from src.save_pickle import save_pickle


@click.command()
@click.option('--raw-data', type=str, help='Path to raw data')
@click.option('--data-to', type=str, help='Path to directory where processed data will be written to')
@click.option('--preprocessor-to', type=str, help='Path to directory where the preprocessor object will be written to')
@click.option('--seed', type=int, help='Random seed', default=123)
def main(raw_data, data_to, preprocessor_to, seed):
    '''
    This script splits the raw data into train and test sets, 
    and then preprocesses the data to be used in exploratory data analysis.
    It also saves the preprocessor to be used in the model training script.
    '''
    np.random.seed(seed)
    set_config(transform_output='pandas')

    # import raw data
    # data located at https://archive.ics.uci.edu/dataset/19/car+evaluation
    colnames = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    car_data = pd.read_csv(raw_data, names=colnames, header=0)

    # train test split, export to csv
    car_train, car_test = train_test_split(
        car_data, train_size=0.8, random_state=522, stratify=car_data['class']
    )

    run_data_validation(car_data)

    car_train.to_csv(os.path.join(data_to, 'car_train.csv'), index=False)
    car_test.to_csv(os.path.join(data_to, 'car_test.csv'), index=False)

    # preprocessing
    # transform categorical features
    car_preprocessor = make_column_transformer(
        (OrdinalEncoder(categories=[['low', 'med', 'high', 'vhigh']]), ['buying']),
        (OrdinalEncoder(categories=[['low', 'med', 'high', 'vhigh']]), ['maint']),
        (OrdinalEncoder(categories=[['2', '3', '4', '5more']]), ['doors']),
        (OrdinalEncoder(categories=[['2', '4', 'more']]), ['persons']),
        (OrdinalEncoder(categories=[['small', 'med', 'big']]), ['lug_boot']),
        (OrdinalEncoder(categories=[['low', 'med', 'high']]), ['safety']),
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    save_pickle(car_preprocessor, preprocessor_to, filename='car_preprocessor.pickle')

    car_preprocessor.fit(car_train)
    encoded_car_train = car_preprocessor.transform(car_train)
    encoded_car_test = car_preprocessor.transform(car_test)

    names = car_preprocessor.get_feature_names_out()
    encoded_car_train = pd.DataFrame(encoded_car_train, columns=names)
    encoded_car_test = pd.DataFrame(encoded_car_test, columns=names)

    encoded_car_train.to_csv(os.path.join(data_to, 'encoded_car_train.csv'), index=False)
    encoded_car_test.to_csv(os.path.join(data_to, 'encoded_car_test.csv'), index=False)


if __name__ == '__main__':
    main()
