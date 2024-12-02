import click
import pickle
import os
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

@click.command()
@click.option('--train_data_from', type=str, help='Path to train data')
@click.option('--target_column', type=str, help='Column name of target in train data')
@click.option('--preprocessor_from', type=str, help='Path to where the preprocessor lives')
@click.option('--results_to', type=str, help='Path to save results to')
@click.option('--set_seed', type=int, help='(Optional) Random seed', default=123)
def main(train_data_from, 
         target_column,
         preprocessor_from, 
         results_to, 
         set_seed):
    """
    Evaluates different models on the train data
    and saves the evaluation results.
    """
    # read in data and preprocessor object
    train_df = pd.read_csv(train_data_from)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    with open(preprocessor_from, 'rb') as f:
        preprocessor = pickle.load(f)

    # dictionary of ML models to evaluate
    models = {
        "Dummy": DummyClassifier(random_state=set_seed),
        "Decision Tree": DecisionTreeClassifier(random_state=set_seed),
        "KNN": KNeighborsClassifier(),
        "SVM RBF": SVC(random_state=set_seed),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=set_seed)
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
        cv_results[model_name] = {
            "mean_train_score": np.mean(scores['train_score']),
            "std_train_score": np.std(scores['train_score']),
            "mean_test_score": np.mean(scores['test_score']),
            "std_test_score": np.std(scores['test_score'])
        }
    
    # save and export results
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(os.path.join(results_to, "model_selection_results.csv"), index=False)

if __name__ == '__main__':
    main()
