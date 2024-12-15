# evaluate_car_predictor.py
# author: Ximin Xu
# date: 2024-12-02

import click
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn import set_config
import sys
from sklearn.model_selection import cross_val_predict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_data import load_data
from src.load_pickle import load_pickle
from src.save_table import save_table
from src.save_plot import save_plot


@click.command()
@click.option('--train-data', type=str, help="Path to train data")
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--pipeline-from', type=str, help="Path to the pipeline pickle")
@click.option('--results-to', type=str, help="Path to directory where the table will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
# Derived from https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/fit_breast_cancer_classifier.py
def main(train_data, test_data, pipeline_from, results_to, plot_to, seed):
    '''Evaluates the car evaluation classifier on the test data
    and saves the evaluation results.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")
    car_test = load_data(test_data)
    car_fit = load_pickle(pipeline_from)
    car_train = load_data(train_data)
    X_test = car_test.drop(columns=["class"])
    X_train = car_train.drop(columns=["class"])
    y_test = car_test["class"]
    y_train = car_train["class"]
    y_pred = car_fit.predict(X_test)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.reset_index(inplace=True)
    class_report_df.rename(columns={"index": "class"}, inplace=True)
    f1 = f1_score(y_test, y_pred, average='weighted')
    test_scores = pd.DataFrame({'f1_score': [f1]})
    save_table(test_scores, results_to, "test_scores.csv")
    save_table(class_report_df, results_to, "classification_report.csv")

    disp = ConfusionMatrixDisplay.from_predictions(y_train, cross_val_predict(car_fit, X_train, y_train))
    figure = disp.figure_
    disp.plot()
    save_plot(figure, plot_to, "confusion_matrix.png")


if __name__ == '__main__':
    main()
