import os
import pytest
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_data import load_data


@pytest.fixture
def setup_csv_file():
    """Fixture to create a temporary CSV file for testing."""
    directory = "./test_dir"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "test_data.csv")
    data = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "Age": [25, 30, 35],
        "City": ["Vancouver", "Kelowna", "Toronto"]
    })
    data.to_csv(file_path, index=False)
    yield file_path, data


def test_load_valid_csv(setup_csv_file):
    """Test loading a valid CSV file."""
    file_path, expected_data = setup_csv_file
    loaded_data = load_data(file_path)
    pd.testing.assert_frame_equal(loaded_data, expected_data, "Loaded data does not match the expected data.")


def test_file_not_exist():
    """Test loading from a non-existent file."""
    non_existent_file = "./nonexistent_dir/nonexistent_file.csv"
    with pytest.raises(ValueError, match="The file provided does not exist."):
        load_data(non_existent_file)


def test_invalid_file_extension(setup_csv_file):
    """Test loading a file with an invalid extension."""
    directory = "./test_dir"
    invalid_file = os.path.join(directory, "invalid_file.txt")
    with open(invalid_file, "w") as file:
        file.write("This is not a CSV file.")

    with pytest.raises(ValueError, match="The file provided is not a CSV file."):
        load_data(invalid_file)
