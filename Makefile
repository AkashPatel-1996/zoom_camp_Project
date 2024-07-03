install:
	pip install --upgrade pip &&\
		pip install -r Code/end_point_code/requirments.txt

test:
	python -m pytest -vv Code/test_*.py

format:
	black *.py


all: install  test