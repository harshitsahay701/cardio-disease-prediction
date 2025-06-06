from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm_model(X_train, y_train, X_test, y_test):
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    print("SVM Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    return model
