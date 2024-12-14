import numpy as np

def tidy_cv_result(model_name, scores):
    '''
    Summarises cross-validation results for a given model.

    -----------
    Parameters:
    -----------
        model_name (str): The name of the model.
        scores (dict): A dictionary that is obtained from the output of `cross_validate` from sklearn.model_selection with argument `return_train_score=True`.

    --------
    Returns:
    --------
        dict: A dictionary with the model name, mean and standard deviation of train scores, and mean and standard deviation of cross-validation scores.

    --------
    Example:
    --------
    from sklearn.model_selection import cross_validate
    scores = cross_validate(
        Ridge(), X_train, y_train, return_train_score=True
    )
    >> cv_results['Ridge'] = tidy_cv_result('Ridge', scores)

    '''
    if not isinstance(model_name, str):
        raise TypeError('The provided model name is not of type `str`.')

    if not isinstance(scores, dict):
        raise TypeError('The provided data is not of type `dict`.')
    
    if not all(key in scores for key in ['train_score', 'test_score']):
        raise ValueError('The dictionary must contain `train_score` and `test_score` keys.')
    
    if not scores['train_score']:
        raise ValueError('`scores` do not contain expected values')
    
    if not scores['test_score']:
        raise ValueError('`scores` do not contain expected values')
    
    result = {
            'Model': model_name,
            'Mean train score': np.mean(scores['train_score']),
            'SD train score': np.std(scores['train_score']),
            'Mean CV score': np.mean(scores['test_score']),
            'SD CV score': np.std(scores['test_score'])
    }

    return result
