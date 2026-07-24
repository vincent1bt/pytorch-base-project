## Base Code to train Pytorch models. Including testing.

Blog post related to this repository  about good code practices to train PyTorch models available [here](https://vincentblog.link/posts/testing-and-workflow-to-train-pytorch-models)

Available Commands:

```
make test-blocks
make test-model
make test-data-loader
make test-train

coverage run --omit='test/*' -m unittest discover test

notebookreader

python -m unittest discover test

coverage report
coverage html
```

#### Other commands

```
uv pip compile pyproject.toml --no-deps -o requirements.txt
```

## TO DO
Share the library to translate code from Jupyter Notebooks to Files/folders.