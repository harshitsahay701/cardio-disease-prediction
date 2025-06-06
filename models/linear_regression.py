from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("LR Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    return model
