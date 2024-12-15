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

This analysis aims to identify the key factors that determine whether a car is considered acceptable, good, or unacceptable according to standardized criteria. To achieve this, several common machine learning models were explored. The SVM RBF classifier was identified as the best-performing model, achieving a test accuracy of 0.952. The SVM RBF model demonstrated exceptional performance, with an accuarcy of 0.99 on unseen data. This makes the SVM RBF model a solid choice for this project. 

The dataset used in this project is the Car Evaluation Database created by M. Bohanec and V. Rajkovic in the early 1990s. 
It was sourced from the UCI Machine Learning Repository and is publicly available for research and can be found in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/19/car+evaluation).

## Report

The final report can be found [here](https://ubc-mds.github.io/Car_Evaluation_Analysis/report/car_evaluation_analysis.html).

## Project Dependencies

* Docker

## Usage

### Setup

1. If you are using Windows or Mac, make sure Docker Desktop is running.

2. Clone this GitHub repository.

### Running the Analysis

1. Navigate to the root of this repository.

2. Run the following command-line command:

    ```bash
    docker compose up
    ```

3. In the terminal, look for a URL that starts with `http://127.0.0.1:8888/lab?token=` as shown in the image below. Copy and paste that URL into your browser. This will launch Jupyter Lab.

    ![jupyter-container-web-app-launch-url](./img/jupyter-container-web-app-launch-url.png)

4. To reset the project to a clean slate, open a terminal in the root of the project and run the following command.

    ```bash
    make clean
    ```

5. To run the entire analysis, open a terminal in the root of the project and run the following command.

    ```bash
    make all
    ```

### Clean Up

1. To shut down the container and clean up the resources, type `Ctrl` + `C` in the terminal where you launched the container. Then, type `docker compose rm`.

## Developer Notes

### Developer Dependencies

* `conda` (version 23.9.0 or higher)
* `conda-lock` (version 2.5.7 or higher)
* Python and packages listed in `environment.yml`

### Adding a New Dependency

1. Create a new branch.

2. Add dependency to the `environment.yaml` file. Make sure to pin the desired package version.

3. Update the `conda-linux-64.yml` by running the following command-line command from the root of the project:

    ```bash
    conda-lock -k explicit --file environment.yml -p linux-64
    ```

4. Rebuild the Docker image locally to ensure it builds and runs properly.

    ```bash
    docker build --tag env-test --platform=linux/amd64 .
    docker run --rm -it --platform=linux/amd64 env-test /bin/bash
    ```

5. Push the changes to GitHub. A new Docker image will be built and pushed to DockerHub automatically. It will be tagged with the SHA for the commit that changed the file.

6. Update the `docker-compose.yaml` file on your branch to use the new container image (make sure to update the tag specifically).

7. Send a Pull Request and merge your branch changes into the `main` branch.

### Running the Test Suite

1. Launch Jupyter Lab using the same `docker compose up` command in the [Running the Analysis](#running-the-analysis) section.

2. To run the test suite, open a terminal in the root of the project and run the `pytest` command.

## License

This dataset is licensed under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.The software code contained within this repository is licensed under the MIT license. See the [license file](https://github.com/UBC-MDS/Car_Evaluation_Analysis/blob/main/LICENSE)for more information.

## References

* Van Rossum, G., & Drake, F. L. (2009). *Python 3 Reference Manual*. CreateSpace, Scotts Valley, CA. ISBN: 1441412697.
* McKinney, W. (2010). Data Structures for Statistical Computing in Python. In *Proceedings of the 9th Python in Science Conference* (pp. 56-61). Edited by S. van der Walt and J. Millman. DOI: [10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a).
* Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95. DOI: [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).
* Pedregosa, F. _et al_. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
* Bohanec, M. (1988). Car Evaluation. UCI Machine Learning Repository. DOI: [10.24432/C5JP48](https://doi.org/10.24432/C5JP48).
* Harris, C. R. _et al_. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362. DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
