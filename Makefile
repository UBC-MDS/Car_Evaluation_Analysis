.PHONY: clean

all: data/raw/car_data_raw.csv \
	data/processed/car_train.csv \
	data/processed/car_test.csv \
	data/processed/encoded_car_train.csv \
	data/processed/encoded_car_test.csv \
	results/models/car_preprocessor.pickle\
	results/figures/feature_counts_by_class.png

data/raw/car_data_raw.csv: scripts/download_data.py
	python scripts/download_data.py \
		--data-to data/raw

data/processed/car_train.csv data/processed/car_test.csv data/processed/encoded_car_train.csv data/processed/encoded_car_test.csv results/models/car_preprocessor.pickle: scripts/split_n_preprocess.py data/raw/car_data_raw.csv
	python scripts/split_n_preprocess.py \
		--raw-data data/raw/car_data_raw.csv \
		--data-to data/processed \
		--preprocessor-to results/models \
		--seed 123
results/figures/feature_counts_by_class.png: scripts/eda.py data/processed/car_train.csv
	python scripts/eda.py \
		--processed-training-data=data/processed/car_train.csv \
		--plot-to=results/figures

clean:
	rm -f data/raw/car_data_raw.csv
	rm -f data/processed/car_train.csv \
		data/processed/car_test.csv \
		data/processed/encoded_car_train.csv \
		data/processed/encoded_car_test.csv \
		results/models/car_preprocessor.pickle \
		results/figures/feature_counts_by_class.png
