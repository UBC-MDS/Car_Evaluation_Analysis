import os
import pytest
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_plot import save_plot

@pytest.fixture
def setup_directory():
    """Fixture to create a temporary directory for testing."""
    directory = "./test_dir"
    os.makedirs(directory, exist_ok=True)
    yield directory

@pytest.fixture
def sample_plot():
    """Fixture to create a sample matplotlib plot."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4], label="tests")
    ax.legend()
    yield fig
    plt.close(fig)

def test_save_valid_plot(setup_directory, sample_plot):
    """Test saving a valid plot to an existing directory."""
    directory = setup_directory
    filename = "test_plot.png"
    save_plot(sample_plot, directory, filename)
    
    filepath = os.path.join(directory, filename)
    assert os.path.exists(filepath), "The plot file was not created."
    assert filepath.endswith(".png"), "The saved file does not have a '.png' extension."

def test_directory_not_exist(sample_plot):
    """Test saving a plot to a non-existent directory."""
    invalid_directory = "./nonexistent_dir"
    with pytest.raises(FileNotFoundError, match=f"The directory '{invalid_directory}' does not exist."):
        save_plot(sample_plot, invalid_directory)


