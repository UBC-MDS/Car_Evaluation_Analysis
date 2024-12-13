import os
import pytest
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_table import save_table

@pytest.fixture
def setup_directory():
    """Fixture to create a temporary directory for testing."""
    directory = "./test_dir"
    os.makedirs(directory, exist_ok=True)
    yield directory

@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame."""
    data = {
        "Name": ["Alice", "Bob", "Charlie"],
        "Score": [95, 85, 75]
    }
    return pd.DataFrame(data)

def test_save_valid_table(setup_directory, sample_dataframe):
    """Test saving a valid DataFrame to a CSV file in an existing directory."""
    directory = setup_directory
    table_name = "test_scores.csv"
    save_table(sample_dataframe, directory, table_name)
    
    file_path = os.path.join(directory, table_name)
    assert os.path.exists(file_path), "The CSV file was not created."
    
    loaded_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe, "Saved DataFrame does not match the original DataFrame.")

def test_directory_not_exist(sample_dataframe):
    """Test saving a DataFrame to a non-existent directory."""
    invalid_directory = "./nonexistent_dir"
    table_name = "test_scores.csv"
    with pytest.raises(FileNotFoundError, match=f"The directory '{invalid_directory}' does not exist."):
        save_table(sample_dataframe, invalid_directory, table_name)
