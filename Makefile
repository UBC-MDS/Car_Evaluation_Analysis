.PHONY: all clean fit evaluate render


all: data/raw/car_data_raw.csv \
    data/processed/car_train.csv \
    data/processed/car_test.csv \
    data/processed/encoded_car_train.csv \
    data/processed/encoded_car_test.csv \
    results/models/car_analysis.pickle \
	results/figures/feature_counts_by_class.png \
	results/tables/model_selection_results.csv \
    results/figures/car_hyperparameter.png \
    results/tables/test_scores.csv \
    results/models/car_preprocessor.pickle \
    report/car_evaluation_analysis.html \
    report/car_evaluation_analysis.pdf

# Download raw data
data/raw/car_data_raw.csv: scripts/download_data.py
	python scripts/download_data.py \
		--data-to data/raw

# Preprocess and split data
data/processed/car_train.csv data/processed/car_test.csv data/processed/encoded_car_train.csv data/processed/encoded_car_test.csv results/models/car_preprocessor.pickle: scripts/split_n_preprocess.py data/raw/car_data_raw.csv
	python scripts/split_n_preprocess.py \
		--raw-data data/raw/car_data_raw.csv \
		--data-to data/processed \
		--preprocessor-to results/models \
		--seed 123

# Generate EDA plots
results/figures/feature_counts_by_class.png: scripts/eda.py data/processed/car_train.csv
	python scripts/eda.py \
		--processed-training-data=data/processed/car_train.csv \
		--plot-to=results/figures

# Select ML models
results/tables/model_selection_results.csv: scripts/select_ml_model.py data/processed/car_train.csv results/models/car_preprocessor.pickle
	python scripts/select_ml_model.py \
		--train-data-from data/processed/car_train.csv \
		--preprocessor-from results/models/car_preprocessor.pickle \
		--results-to results/tables

# Fit the model and generate hyperparameter tuning results
results/models/car_analysis.pickle results/figures/car_hyperparameter.png: scripts/fit_car_analysis_classifier.py data/processed/car_train.csv results/models/car_preprocessor.pickle
	python scripts/fit_car_analysis_classifier.py \
		--training-data data/processed/car_train.csv \
		--preprocessor results/models/car_preprocessor.pickle \
		--pipeline-to results/models \
		--plot-to results/figures \
		--seed 123

# Evaluate the model and save test scores
results/tables/test_scores.csv: scripts/evaluate_car_predictor.py data/processed/car_test.csv results/models/car_analysis.pickle
	python scripts/evaluate_car_predictor.py \
		--test-data data/processed/car_test.csv \
		--pipeline-from results/models/car_analysis.pickle \
		--results-to results/tables \
		--seed 123

# Render the Quarto report
report/car_evaluation_analysis.html: report/car_evaluation_analysis.qmd results/tables/test_scores.csv results/models/car_analysis.pickle
	quarto render report/car_evaluation_analysis.qmd --to html

report/car_evaluation_analysis.pdf: report/car_evaluation_analysis.qmd results/tables/test_scores.csv results/models/car_analysis.pickle
	quarto render report/car_evaluation_analysis.qmd --to pdf

# Clean up generated files
clean:
	rm -f data/raw/car_data_raw.csv
	rm -f data/processed/car_train.csv \
		data/processed/car_test.csv \
		data/processed/encoded_car_train.csv \
		data/processed/encoded_car_test.csv
	rm -f results/models/car_preprocessor.pickle \
		results/models/car_analysis.pickle
	rm -f results/figures/car_hyperparameter.png \
		results/figures/feature_counts_by_class.png
	rm -f results/tables/model_selection_results.csv
	rm -f results/tables/test_scores.csv
	rm -f report/car_evaluation_analysis.html \
		report/car_evaluation_analysis.pdf
