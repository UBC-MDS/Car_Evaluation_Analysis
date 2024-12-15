import os
import pickle
import pytest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_pickle import load_pickle


@pytest.fixture
def setup_pickle_file():
    """Fixture to create a temporary pickle file for testing."""
    directory = "./test_dir"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "test_object.pickle")
    obj = {"key": "value"}
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)
    yield file_path, obj


def test_load_valid_pickle(setup_pickle_file):
    """Test loading a valid pickle file."""
    file_path, expected_obj = setup_pickle_file
    loaded_obj = load_pickle(file_path)
    assert loaded_obj == expected_obj, "Loaded object does not match the saved object."


def test_file_not_exist():
    """Test loading from a non-existent file."""
    non_existent_file = "./nonexistent_dir/nonexistent_file.pickle"
    with pytest.raises(ValueError, match="The file provided does not exist."):
        load_pickle(non_existent_file)


def test_invalid_file_extension(setup_pickle_file):
    """Test loading a file with an invalid extension."""
    directory = "./test_dir"
    invalid_file = os.path.join(directory, "invalid_file.txt")
    with open(invalid_file, "w") as file:
        file.write("This is not a pickle file.")

    with pytest.raises(ValueError, match="The file provided is not a pickle file."):
        load_pickle(invalid_file)


def test_invalid_pickle_content(setup_pickle_file):
    """Test loading an invalid pickle file."""
    directory = "./test_dir"
    invalid_pickle_file = os.path.join(directory, "invalid_pickle.pickle")
    with open(invalid_pickle_file, "wb") as file:
        file.write(b"This is not a valid pickle content.")

    with pytest.raises(ValueError, match="Error loading the pickle file:"):
        load_pickle(invalid_pickle_file)
