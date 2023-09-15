# python-poetry-template

This is a template repository for a python project using poetry.

## Setup

Rename <sample_lib> into your package name in 

- `/sample_lib` (directory name)
- `/Makefile`
- `pyproject.toml`.

After that, by `make build` command, `setup.py` can be created.
Then you can install this package via pip by 

```shell
pip install git+https://github.com/<user_id>/<package_name>
```


## Commands

### Create `setup.py`

```shell
make build
```


### formatting

```shell
make format 
```

### lint
```shell
make lint
```

### test

```shell
make test
```



