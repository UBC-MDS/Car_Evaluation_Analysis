.PHONY: all clean fit evaluate

all: data/raw/car_data_raw.csv \
	data/processed/car_train.csv \
	data/processed/car_test.csv \
	data/processed/encoded_car_train.csv \
	data/processed/encoded_car_test.csv \
	results/models/car_preprocessor.pickle \
	results/models/car_analysis.pickle \
	results/figures/car_hyperparameter.png \
	results/tables/test_scores.csv

data/raw/car_data_raw.csv: scripts/download_data.py
	python scripts/download_data.py \
		--data-to data/raw

data/processed/car_train.csv data/processed/car_test.csv data/processed/encoded_car_train.csv data/processed/encoded_car_test.csv results/models/car_preprocessor.pickle: scripts/split_n_preprocess.py data/raw/car_data_raw.csv
	python scripts/split_n_preprocess.py \
		--raw-data data/raw/car_data_raw.csv \
		--data-to data/processed \
		--preprocessor-to results/models \
		--seed 123

results/models/car_analysis.pickle results/figures/car_hyperparameter.png: scripts/fit_car_analysis_classifier.py data/processed/car_train.csv results/models/car_preprocessor.pickle
	python scripts/fit_car_analysis_classifier.py \
		--training-data data/processed/car_train.csv \
		--preprocessor results/models/car_preprocessor.pickle \
		--pipeline-to results/models \
		--plot-to results/figures \
		--seed 123

results/tables/test_scores.csv: scripts/evaluate_car_predictor.py data/processed/car_test.csv results/models/car_analysis.pickle
	python scripts/evaluate_car_predictor.py \
		--test-data data/processed/car_test.csv \
		--pipeline-from results/models/car_analysis.pickle \
		--results-to results/tables \
		--seed 123

clean:
	rm -f data/raw/car_data_raw.csv
	rm -f data/processed/car_train.csv \
		data/processed/car_test.csv \
		data/processed/encoded_car_train.csv \
		data/processed/encoded_car_test.csv
	rm -f results/models/car_preprocessor.pickle \
		results/models/car_analysis.pickle
	rm -f results/figures/car_hyperparameter.png
	rm -f results/tables/test_scores.csv
