import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config

@click.command()
@click.option('--test-data', type=str, help="Path to scaled test data")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
#Derived from https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/fit_breast_cancer_classifier.py 

def main(test_data, pipeline_from, results_to, seed):
    '''Evaluates the car evaluation classifier on the test data 
    and saves the evaluation results.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")
    car_test = pd.read_csv(test_data)
    with open(pipeline_from, 'rb') as f:
        car_fit = pickle.load(f)
    accuracy = car_fit.score(
        car_test.drop(columns=["class"]),
        car_test["class"]
    )
    test_scores = pd.DataFrame({'accuracy': [accuracy]})
    test_scores.to_csv(os.path.join(results_to, "test_scores.csv"), index=False)
    
if __name__ == '__main__':
    main()