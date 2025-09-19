## Base Code to train Pytorch models. Including testing.

Blog post about good code practices to train PyTorch models related to this repository available [here](https://vincentblog.link/posts/testing-and-workflow-to-train-pytorch-models)

Commands to run testing:

```
python -m unittest -v test/test_blocks.py
python -m unittest -v test/test_model.py
python -m unittest -v test/test_train.py
python -m unittest -v test/test_data_loader.py

coverage run --omit='test/*' -m unittest discover test

notebookreader

python -m unittest discover test

coverage report
coverage html
```

## TO DO
Share the library to translate code from Jupyter Notebooks to Files/folders.