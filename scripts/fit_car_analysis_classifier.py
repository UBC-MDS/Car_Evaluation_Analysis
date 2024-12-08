# fit_car_analysis_classifier.py
# author: Ximin Xu
# date: 2024-12-02

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump
import matplotlib.pyplot as plt

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor pickle")
@click.option('--pipeline-to', type=str, help="Path to directory of preprocessor pickle")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)


def main(training_data, preprocessor, pipeline_to, plot_to, seed):
    '''Fits the car analysis classifier on training data to find out the optimized hyperparameter and save the pipeline
    '''
    np.random.seed(seed)
    set_config(transform_output="pandas")
    car_train = pd.read_csv(training_data)
    car_preprocessor = pickle.load(open(preprocessor, "rb"))

    
    svc = SVC()
    car_pipe = make_pipeline(car_preprocessor, svc)
    param_grid = {
    "svc__gamma": 10.0 ** np.arange(-5, 5, 1),
    "svc__C": 10.0 ** np.arange(-5, 5, 1)
}
    random_search = RandomizedSearchCV(
    car_pipe, param_distributions=param_grid, n_iter=100, n_jobs= -1, return_train_score=True
)
    random_search.fit(car_train.drop(columns = ['class']), car_train['class'] )
    with open(os.path.join(pipeline_to, "car_analysis.pickle"), 'wb') as f:
        pickle.dump(random_search, f)
    results = pd.DataFrame(random_search.cv_results_)

    pivot_table = results.pivot(index="param_svc__gamma", columns="param_svc__C", values="mean_test_score")
    plt.figure(figsize=(7, 6))
    plt.title("Mean Test Score for different Gamma and C values")
    plt.xlabel('C ')
    plt.ylabel('Gamma')
    plt.imshow(pivot_table, cmap='viridis', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Mean Test Score')
    c_values = pivot_table.columns.astype(float)
    gamma_values = pivot_table.index.astype(float)
    plt.xticks(
    ticks=np.arange(len(c_values)), 
    labels=[f"$10^{{{int(np.log10(c))}}}$" for c in c_values],
    rotation=45, ha='right'
    )
    plt.yticks(
    ticks=np.arange(len(gamma_values)), 
    labels=[f"$10^{{{int(np.log10(g))}}}$" for g in gamma_values]
    )
    plt.savefig(os.path.join(plot_to, "car_hyperparameter.png"))

if __name__ == '__main__':
    main()