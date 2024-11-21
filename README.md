# Car Evaluation Analysis

Authors: Danish Karlin Isa, Nicholas Varabioff, Ximin Xu, Zuer Zhong

This project is part of the coursework for DSCI 522 Data Science Workflows, a course of the Master of Data Science program at the University of British Columbia.

## About

This project attempts to predict the level of acceptability of cars using machine learning methods.
Using a 1997 dataset, the influence of various attributes of a new car on its acceptability among customers is analysed.
These attributes include:

* The buying price of the car
* Maintenance costs
* Number of doors
* Passenger capacity
* Boot size
* Safety ratings

This analysis aims to identify the key factors that determine whether a car is considered acceptable, good, or unacceptable according to standardized criteria.

The dataset used in this project is the Car Evaluation Database created by M. Bohanec and V. Rajkovic in the early 1990s. 
It was sourced from the UCI Machine Learning Repository and is publicly available for research and can be found in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/19/car+evaluation).

## Usage
To run this project, follow these steps from the root of this repository:

1. If you are running this project for the first time, run the following from a Command-Line Interface:

```bash
conda-lock install --name Car_Evaluation_Analysis conda-lock_<your_operating_system>.yml
```

2. To run the analysis, run the following command:

```bash
jupyter lab
```

Open `notebooks/Car_Evaluation_Analysis.ipynb` in Jupyter Lab and under Switch/Select Kernel choose "Python [conda env:car_evaluation_analysis]".

Next, under the "Kernel" menu click "Restart Kernel and Run All Cells...".

## Dependencies

* altair=5.1.2
* ipykernel=6.26.0
* pandas=2.1.2
* python=3.11.6
* scikit-learn=1.3.2
* vegafusion=1.4.3
* vegafusion-jupyter=1.4.3
* vegafusion-python-embed=1.4.3
* vl-convert-python=1.0.1
* requests=2.31.0
* notebook=6.5.4
* jupyter_contrib_nbextensions=0.7.0

## License

This dataset is licensed under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.The software code contained within this repository is licensed under the MIT license. See the [license file](https://github.com/UBC-MDS/Car_Evaluation_Analysis/blob/main/LICENSE)for more information.

## References

* Dua, Dheeru, and Casey Graff. 2017. "UCI Machine Learning Repository." University of California, Irvine, School of Information and Computer Sciences. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
* Bohanec, M. (1988). Car Evaluation [Dataset]. UCI Machine Learning Repository. (https://doi.org/10.24432/C5JP48).
