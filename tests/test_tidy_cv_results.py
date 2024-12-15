# test_tidy_cv_results.py
# author: Danish Karlin Isa
# date: 2024-12-13

import pytest
from src.tidy_cv_results import tidy_cv_result


model_name = 'test model'
scores = {
    'train_score': [0.2, 0.4, 0.6],
    'test_score': [0.1, 0.5, 0.9]
}
expected = {
            'Model': 'test model',
            'Mean train score': 0.4,
            'SD train score': 0.16329931618555,
            'Mean CV score': 0.5,
            'SD CV score': 0.32659863237109
}


def test_tidy_cv_result_valid_input():
    result = tidy_cv_result(model_name, scores)
    assert result['Model'] == expected['Model']
    assert result['Mean train score'] == pytest.approx(expected['Mean train score'])
    assert result['SD train score'] == pytest.approx(expected['SD train score'])
    assert result['Mean CV score'] == pytest.approx(expected['Mean CV score'])
    assert result['SD CV score'] == pytest.approx(expected['SD CV score'])


def test_tidy_cv_result_invalid_model_name():
    name_not_str = 522
    with pytest.raises(TypeError):
        tidy_cv_result(name_not_str, scores)


def test_tidy_cv_result_invalid_scores_type():
    scores_not_dict = 'scores'
    with pytest.raises(TypeError):
        tidy_cv_result(model_name, scores_not_dict)


def test_tidy_cv_result_missing_keys():
    scores_invalid_keys = {
        'hello': 522,
        'world': 'mds'
    }
    with pytest.raises(ValueError):
        tidy_cv_result(model_name, scores_invalid_keys)


def test_tidy_cv_result_empty_scores():
    scores_empty = {
        'train_score': [],
        'test_score': []
    }
    with pytest.raises(ValueError):
        tidy_cv_result(model_name, scores_empty)
