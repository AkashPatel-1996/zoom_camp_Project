
import xgboost as xgb
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import mlflow
from sklearn.linear_model import LogisticRegression
from functools import partial
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def objective(params,X,Y):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cancer dataset pipeline experiment")
    with mlflow.start_run():

        mlflow.log_params(params)
        model = LogisticRegression(
            C=params['C'],
            penalty=params['penalty'],
            max_iter=10000,
            solver=params['solver'],
            random_state=42
        )
        X_train, X_test, Y_train, Y_test = train_test_split(X , Y, random_state = 412 , test_size=0.5)
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        print(accuracy)
        
        
        return {'loss': 1 - accuracy, 'status': STATUS_OK}


def parameters(space,X,Y):

    trials = Trials()
    objective_with_data = partial(objective, X=X, Y=Y)

    best_params = fmin(
        fn=objective_with_data,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )
    return best_params




