import os
import pickle
import pytest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_pickle import save_pickle


@pytest.fixture
def test_dir():
    directory = "./test_dir"
    os.makedirs(directory, exist_ok=True)
    yield directory


def test_save_valid_directory(test_dir):
    filename = "test_object.pickle"
    obj = {"key": "value"}
    save_pickle(obj, test_dir, filename)
    filepath = os.path.join(test_dir, filename)
    assert os.path.exists(filepath), "File was not created."
    with open(filepath, "rb") as file:
        loaded_obj = pickle.load(file)
    assert loaded_obj == obj, "Loaded object does not match saved object."


def test_invalid_directory():
    invalid_directory = "./nonexistent_dir"
    obj = {"key": "value"}
    with pytest.raises(FileNotFoundError, match=f"The directory '{invalid_directory}' does not exist."):
        save_pickle(obj, invalid_directory)
