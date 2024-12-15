# test_data_validation.py
# author: Nicholas Varabioff
# date: 2024-12-15
# Code references https://github.com/ttimbers/breast-cancer-predictor/blob/3.0.0/tests/test_validate_data.py

import pytest
import sys
import os
import pandas as pd
import pandera as pa
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_validation import run_data_validation

# Set up valid data
valid_data = pd.DataFrame({
    'buying': ['low', 'med', 'high', 'vhigh'],
    'maint': ['low', 'med', 'high', 'vhigh'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more', '2'],
    'lug_boot': ['small', 'med', 'big', 'small'],
    'safety': ['low', 'med', 'high', 'low'],
    'class': ['unacc', 'acc', 'vgood', 'good']
})

# Test wrong type passed to function
valid_data_as_np = valid_data.copy().to_numpy()
def test_data_validation_type():
    with pytest.raises(TypeError):
        run_data_validation(valid_data_as_np)

# Test empty data frame
empty_data_frame = valid_data.copy().iloc[0:0]
def test_data_validation_empty():
    with pytest.raises(ValueError):
        run_data_validation(empty_data_frame)

# Set up invalid data cases
invalid_data_cases = []

# Test missing class column
missing_class_col = valid_data.copy()
missing_class_col = missing_class_col.drop('class', axis=1)  # drop class column
invalid_data_cases.append((missing_class_col, '`class` from DataFrameSchema'))

# Test wrong levels in class column
wrong_category_label = valid_data.copy()
wrong_category_label.loc[0, 'class'] = 'invalid'
invalid_data_cases.append((wrong_category_label, 'Check absent or incorrect for wrong string value/category in "class" column'))

# Test missing value in class column
missing_class = valid_data.copy()
missing_class.loc[0, 'class'] = None
invalid_data_cases.append((missing_class, 'Check absent or incorrect for missing/null "class" value'))

# Test missing columns (one for each column)
for col in valid_data.columns:
    missing_col = valid_data.copy()
    missing_col = missing_col.drop(col, axis=1)  # drop column
    invalid_data_cases.append((missing_col, f'"{col}" is missing from DataFrameSchema'))

# Test incorrect values in columns (one for each column)
for col in valid_data.columns:
    wrong_type = valid_data.copy()
    wrong_type.loc[0, col] = 'incorrect_val'
    invalid_data_cases.append((wrong_type, f'Check incorrect value for category levels in "{col}" is missing or incorrect'))

# Test duplicate observations
duplicate = valid_data.copy()
duplicate = pd.concat([duplicate, duplicate.iloc[[0], :]], ignore_index=True)
invalid_data_cases.append((duplicate, f'Check absent or incorrect for duplicate rows'))


# Parameterize invalid data test cases
@pytest.mark.parametrize("invalid_data, description", invalid_data_cases)
def test_valid_w_invalid_data(invalid_data, description):
    with pytest.raises(pa.errors.SchemaErrors):
        run_data_validation(invalid_data)
