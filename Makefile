install:
	pip install --upgrade pip &&\
		pip install -r requirments.txt

test:
	python -m pytest -vv Code/test_*.py

format:
	black *.py


all: install  test