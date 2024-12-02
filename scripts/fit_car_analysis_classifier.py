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
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--columns-to-drop', type=str, help="Optional: columns to drop")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
#Derived from https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/fit_breast_cancer_classifier.py 

def main(training_data, preprocessor, pipeline_to, plot_to, seed):
    '''Fits the car analysis classifier on training data to find out the optimized hyperparameter and save the pipeline
    '''
    np.random.seed(seed)
    set_config(transform_output="pandas")
    car_train = pd.read_csv(training_data)
    car_preprocessor = pickle.load(open(preprocessor), "rb")
    
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
    plt.xticks(np.arange(len(pivot_table.columns)), pivot_table.columns)
    plt.yticks(np.arange(len(pivot_table.index)), pivot_table.index)
    plt.savefig(os.path.join(plot_to, "car_hyperparameter.png"))

if __name__ == '__main__':
    main()