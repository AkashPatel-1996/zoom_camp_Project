from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from finding_parameter import parameters
from functools import partial
from sklearn.linear_model import LogisticRegression


def download_data(dataset):
    if dataset == "cancer":
        data = load_breast_cancer()
    return data


def prepare_data(data):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df = pd.concat([df, df], ignore_index=True)
    return df

def feature_extraction(data):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    removed_col = ['target','mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']
    X = df.drop(columns=  removed_col )
    Y = df['target']
    return X , Y

def find_best_model(X, Y):
    space = {
    'C': hp.loguniform('C', np.log(1e-4), np.log(1e2)),
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'solver': hp.choice('solver', ['liblinear', 'saga'])
     }
    best_params = parameters(space,X,Y)
    return best_params 


def train(X,Y,params):
    best_model = LogisticRegression(
    C= params['C'],
    penalty=['l1', 'l2'][params['penalty']],
    max_iter=100000,
    solver='liblinear'
)    
    X_train, X_test, Y_train, Y_test = train_test_split(X , Y, random_state = 412 , test_size=0.5)
    best_model.fit(X_train, Y_train)
    return best_model

    