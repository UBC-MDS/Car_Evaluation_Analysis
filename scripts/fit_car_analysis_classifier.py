# fit_car_analysis_classifier.py
# author: Ximin Xu
# date: 2024-12-02

import click
import os
import numpy as np
import pandas as pd
import sys
from sklearn import set_config
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_pickle import save_pickle
from src.load_data import load_data
from src.load_pickle import load_pickle
from src.save_plot import save_plot
import matplotlib.pyplot as plt

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor pickle")
@click.option('--pipeline-to', type=str, help="Path to directory to save pipeline")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(training_data, preprocessor, pipeline_to, plot_to, seed):
    '''Fits the car analysis classifier on training data to find out the optimized hyperparameter and save the pipeline'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    car_train = load_data(training_data)
    car_preprocessor = load_pickle(preprocessor)

    svc = SVC()
    car_pipe = make_pipeline(car_preprocessor, svc)

    param_grid = {
        "svc__gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "svc__C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    }

    random_search = RandomizedSearchCV(
        car_pipe, param_distributions=param_grid, n_iter=100, n_jobs=-1, scoring = 'f1', return_train_score=True
    )
    random_search.fit(car_train.drop(columns=['class']), car_train['class'])

    save_pickle(random_search, pipeline_to, filename="car_analysis.pickle")

    results = pd.DataFrame(random_search.cv_results_)
    pivot_table = results.pivot(index="param_svc__gamma", columns="param_svc__C", values="mean_test_score")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Mean Test Score for different Gamma and C values")
    ax.set_xlabel('C ')
    ax.set_ylabel('Gamma')
    cax = ax.imshow(pivot_table, cmap='viridis', interpolation='nearest', aspect='auto')
    fig.colorbar(cax, label='Mean Test Score')
    c_values = pivot_table.columns.astype(float)
    gamma_values = pivot_table.index.astype(float)
    ax.set_xticks(np.arange(len(c_values)))
    ax.set_xticklabels([f"$10^{{{int(np.log10(c))}}}$" for c in c_values], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(gamma_values)))
    ax.set_yticklabels([f"$10^{{{int(np.log10(g))}}}$" for g in gamma_values])

    save_plot(fig, plot_to, filename="car_hyperparameter.png")

if __name__ == '__main__':
    main()
