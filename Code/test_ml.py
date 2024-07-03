import pytest
import pandas as pd
from sklearn.datasets import load_breast_cancer
from Pipeline_functions import download_data, prepare_data, feature_extraction, find_best_model, train

@pytest.fixture
def data():
    return download_data("cancer")

@pytest.fixture
def prepared_data(data):
    return prepare_data(data)

@pytest.fixture
def features(data):
    return feature_extraction(data)

def test_download_data(data):
    assert data is not None
    assert len(data.data) == 569

def test_prepare_data(prepared_data, data):
    assert isinstance(prepared_data, pd.DataFrame)
    assert prepared_data.shape[1] == len(data.feature_names)
    assert prepared_data.shape[0] == 2 * len(data.data)

def test_feature_extraction(features):
    X, Y = features
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.Series)
    assert X.shape[1] == 4
    assert len(Y) == len(X)

def test_find_best_model(features):
    X, Y = features
    best_params = find_best_model(X, Y)
    assert 'C' in best_params
    assert 'penalty' in best_params
    assert 'solver' in best_params


if __name__ == "__main__":
    pytest.main()
