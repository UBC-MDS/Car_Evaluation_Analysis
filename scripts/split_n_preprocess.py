# split_n_preprocess.py
# author: Nicholas Varabioff
# date: 2024-02-12

import click
import os
import pickle
import numpy as np
import pandas as pd
import pandera as pa
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


@click.command()
@click.option('--raw-data-dir', type=str, help="Path to raw data directory")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)


def main(raw_data, data_to, preprocessor_to, seed):
    '''
    This script splits the raw data into train and test sets, 
    and then preprocesses the data to be used in exploratory data analysis.
    It also saves the preprocessor to be used in the model training script.
    '''
    np.random.seed(seed)
    set_config(transform_output="pandas")
    
    # import raw data
    # data located at https://archive.ics.uci.edu/dataset/19/car+evaluation
    colnames = ['buying','maint','doors','persons','lug_boot','safety','class']
    car_data = pd.read_csv(raw_data, names=colnames, header=None)
    
    # Validate data schema with Pandera
    # Correct data types in each column
    # No duplicate observations,
    # No outlier or anomalous values, since all of our data are categorical features, no need for this
    schema = pa.DataFrameSchema(
        {
            'buying': pa.Column(str, pa.Check.isin(['low','med','high','vhigh']), nullable=False),
            'maint': pa.Column(str, pa.Check.isin(['low','med','high','vhigh']), nullable=False),
            'doors': pa.Column(str, pa.Check.isin(['2','3','4','5more']), nullable=False),
            'persons': pa.Column(str, pa.Check.isin(['2','4','more']), nullable=False),
            'lug_boot': pa.Column(str, pa.Check.isin(['small','med','big']), nullable=False),
            'safety': pa.Column(str, pa.Check.isin(['low','med','high']), nullable=False),
            'class': pa.Column(str, pa.Check.isin(['unacc','acc','vgood','good']), nullable=False)
        },
        checks=[
            pa.Check(lambda car_data: ~car_data.duplicated().any(), error='Duplicate rows found.')
        ]
    )
    schema.validate(car_data, lazy=True)

    # train test split, export to csv
    car_train, car_test = train_test_split(
        car_data, train_size=0.8, random_state=522, stratify=car_data['class']
    )

    car_train.to_csv(os.path.join(data_to, "car_train.csv"), index=False)
    car_test.to_csv(os.path.join(data_to, "car_test.csv"), index=False)

    # preprocessing
    # transform categorical features
    car_preprocessor = make_column_transformer(
        (OrdinalEncoder(categories=[['low','med','high','vhigh']]), ['buying']),
        (OrdinalEncoder(categories=[['low','med','high','vhigh']]), ['maint']),
        (OrdinalEncoder(categories=[['2','3','4','5more']]), ['doors']),
        (OrdinalEncoder(categories=[['2','4','more']]), ['persons']),
        (OrdinalEncoder(categories=[['small','med','big']]), ['lug_boot']),
        (OrdinalEncoder(categories=[['low','med','high']]), ['safety']),
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    pickle.dump(car_preprocessor, open(os.path.join(preprocessor_to, "car_preprocessor.pickle"), "wb"))
    
    car_preprocessor.fit(car_train)
    encoded_car_train = car_preprocessor.transform(car_train)
    encoded_car_test = car_preprocessor.transform(car_test)

    names = car_preprocessor.get_feature_names_out()
    encoded_car_train = pd.DataFrame(encoded_car_train, columns=names)
    encoded_car_test = pd.DataFrame(encoded_car_test, columns=names)

    encoded_car_train.to_csv(os.path.join(data_to, "encoded_car_train.csv"), index=False)
    encoded_car_test.to_csv(os.path.join(data_to, "encoded_car_test.csv"), index=False)


if __name__ == '__main__':
    main()