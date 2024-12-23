## Testing temp readme file

```
python -m unittest -v test/test_blocks.py

python -m unittest -v test/test_model.py

python -m unittest -v test/test_data_loader.py

coverage run --omit='test/*' -m unittest discover test

notebookreader

python -m unittest discover test

coverage report
coverage html
```