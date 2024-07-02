
import xgboost as xgb
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import mlflow


def objective(params):
    with mlflow.start_run():
        mlflow.log_params(params)
        model = LogisticRegression(
            C=params['C'],
            penalty=params['penalty'],
            max_iter=10000,
            solver=params['solver'],
            random_state=42
        )
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        print(accuracy)
        
        return {'loss': 1 - accuracy, 'status': STATUS_OK}


def parameters(space):

    trials = Trials()

    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )
    return best_params

def train(X,Y,params):
    best_model = LogisticRegression(
    C=111111,
    penalty=['l1', 'l2'][params['penalty']],
    max_iter=100000,
    solver='liblinear'
)
    best_model.fit(X_train, Y_train)


