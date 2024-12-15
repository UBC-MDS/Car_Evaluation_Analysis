import os
import pytest
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.edafunction import save_altair_plot, create_target_distribution_plot, create_feature_distributions_plot


@pytest.fixture
def setup_directory():
    """
    Fixture to create a temporary directory for testing.
    Keeps the directory and its contents after the test.
    """
    directory = "./tests/test_dir" 
    os.makedirs(directory, exist_ok=True)
    yield directory
    print(f"Test files saved in: {directory}")


@pytest.fixture
def raw_data_sample():
    """
    Provides a sample raw data DataFrame for testing.
    """
    return pd.DataFrame({
        "class": ["acc", "unacc", "good", "vgood", "unacc", "acc"]
    })


@pytest.fixture
def processed_data_sample():
    """
    Provides a sample processed data DataFrame for testing.
    """
    return pd.DataFrame({
        "buying": ["low", "med", "high", "vhigh", "low", "med"],
        "maint": ["low", "med", "high", "vhigh", "low", "med"],
        "doors": ["2", "3", "4", "5more", "2", "3"],
        "persons": ["2", "4", "more", "4", "2", "4"],
        "lug_boot": ["small", "big", "med", "big", "small", "big"],
        "safety": ["low", "high", "med", "high", "low", "high"],
        "class": ["acc", "unacc", "good", "vgood", "unacc", "acc"]
    })


def test_create_target_distribution_plot(setup_directory, raw_data_sample):
    """
    Test creating and saving a target distribution plot.
    """
    directory = setup_directory
    filename = "target_distribution_plot.png"

    plot = create_target_distribution_plot(raw_data_sample)
    save_altair_plot(plot, directory, filename)

    filepath = os.path.join(directory, filename)
    assert os.path.exists(filepath), "The target distribution plot file was not created."


def test_create_feature_distributions_plot(setup_directory, processed_data_sample):
    """
    Test creating and saving a feature distributions plot.
    """
    directory = setup_directory
    filename = "feature_distributions_plot.png"
    features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

    plot = create_feature_distributions_plot(processed_data_sample, features)
    save_altair_plot(plot, directory, filename)

    filepath = os.path.join(directory, filename)
    assert os.path.exists(filepath), "The feature distributions plot file was not created."
