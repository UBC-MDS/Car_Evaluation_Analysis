# evaluate_models.py
# author: Danish Karlin Isa
# date: 2024-12-02

import click
import os
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_data import load_data
from src.load_pickle import load_pickle
from src.save_table import save_table
from src.tidy_cv_results import tidy_cv_result


@click.command()
@click.option('--train-data-from', type=str, help='Path to train data')
@click.option('--preprocessor-from', type=str, help='Path to where the preprocessor lives')
@click.option('--results-to', type=str, help='Path to save results to')
@click.option('--set-seed', type=int, help='(Optional) Random seed', default=123)
def main(train_data_from,
         preprocessor_from,
         results_to,
         set_seed):
    """
    Evaluates different models on the train data
    and saves the evaluation results.
    """
    # read in data and preprocessor object
    train_df = load_data(train_data_from)

    target_column = 'class'
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    preprocessor = load_pickle(preprocessor_from)

    # dictionary of ML models to evaluate
    models = {
        "Dummy": DummyClassifier(random_state=set_seed),
        "Decision Tree": DecisionTreeClassifier(random_state=set_seed),
        "KNN": KNeighborsClassifier(),
        "SVM RBF": SVC(random_state=set_seed),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=set_seed)
    }

    # df to store cv results
    cv_results = {}

    # evaluate train data on every model in models
    for model_name, model in models.items():
        pipe = make_pipeline(
            preprocessor,
            model
        )

        scores = cross_validate(
            pipe, X_train, y_train, n_jobs=-1, 
            return_train_score=True, cv=5
        )

        cv_results[model_name] = tidy_cv_result(model_name, scores)

    cv_results_df = pd.DataFrame(cv_results).T
    save_table(cv_results_df, results_to, "model_selection_results.csv")


if __name__ == '__main__':
    main()
