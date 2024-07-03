install:
	pip install --upgrade pip &&\
		pip install -r requirments.txt

test:
	python -m pytest -vv test_*.py

format:
	black *.py


all: install lint test