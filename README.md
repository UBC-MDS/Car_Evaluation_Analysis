# Car_Evaluation_Analysis
## Authors
- Danish Karlin Isa, Nicholas Varabioff, XiMin Xu & Zuer Zhong
### This project is a demonstration of data analysis workflows for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia.
## About
- Our project investigates the prediction of  car acceptability using machine learning methods. Using a dataset spanning from 1997, we analyze the influence of various attributes such as buying price, maintenance cost, number of doors, person capacity, luggage boot size, and safety ratings on the overall acceptability of cars. This analysis can identify the key factors that determine whether a car is considered acceptable, good, or unacceptable according to standardized criteria.

- The dataset that was used in this project is of Car Evaluation Database created by the efforts of M. Bohanec and V. Rajkovic in the early 1990s. It was sourced from the UCI Machine Learning Repository and is publicly available for research and can be found here [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/19/car+evaluation).Each row in the dataset details a car’s attributes, including its buying price, maintenance cost, number of doors, capacity to seat persons, luggage boot size, and safety rating. 
## Report
- The final report can be found here.
## Usage
To run this project, follow these steps from the root of this repository:
1. First time running the project, run the following from the root of this repository:
   ```bash
   conda-lock install --name Car_Evaluation_Analysis conda-lock.yml
2. To run the analysis, run the following from the root of this repository:
   ```bash
   jupyter lab
Open notebooks/ Car_Evaluation_Analysis.ipynb in Jupyter Lab and under Switch/Select Kernel choose "Python [conda env:car_evaluation_analysis]".

Next, under the "Kernel" menu click "Restart Kernel and Run All Cells...".
## Dependencies
  - altair=5.1.2
  - ipykernel=6.26.0
  - pandas=2.1.2
  - python=3.11.6
  - scikit-learn=1.3.2
  - vegafusion=1.4.3
  - vegafusion-jupyter=1.4.3
  - vegafusion-python-embed=1.4.3
  - vl-convert-python=1.0.1
  - requests=2.31.0
  - notebook=6.5.4
  - jupyter_contrib_nbextensions=0.7.0
## License
This dataset is licensed under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.The software code contained within this repository is licensed under the MIT license. See the [license file](https://github.com/UBC-MDS/Car_Evaluation_Analysis/blob/main/LICENSE)for more information.
## References
- Dua, Dheeru, and Casey Graff. 2017. "UCI Machine Learning Repository." University of California, Irvine, School of Information and Computer Sciences. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
- Bohanec, M. (1988). Car Evaluation [Dataset]. UCI Machine Learning Repository. (https://doi.org/10.24432/C5JP48).
