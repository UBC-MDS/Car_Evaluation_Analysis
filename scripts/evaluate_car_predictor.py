# evaluate_car_predictor.py
# author: Ximin Xu
# date: 2024-12-02

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_data import load_data
from src.load_pickle import load_pickle
from src.save_table import save_table

@click.command()
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--pipeline-from', type=str, help="Path to the pipeline pickle")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
#Derived from https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/fit_breast_cancer_classifier.py 

def main(test_data, pipeline_from, results_to, seed):
    '''Evaluates the car evaluation classifier on the test data 
    and saves the evaluation results.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")
    car_test = load_data(test_data)
    car_fit = load_pickle(pipeline_from)
    accuracy = car_fit.score(
        car_test.drop(columns=["class"]),
        car_test["class"]
    )
    test_scores = pd.DataFrame({'accuracy': [accuracy]})
    save_table(test_scores, results_to, "test_scores.csv")
    
if __name__ == '__main__':
    main()