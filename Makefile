install:
	pip install --upgrade pip && pip install -r requirements.txt 

test-blocks:
	python -m unittest -v test/test_blocks.py

test-model:
	python -m unittest -v test/test_model.py

test-data-loader:
	python -m unittest -v test/test_data_loader.py

test-train:
	python -m unittest -v test/test_train.py