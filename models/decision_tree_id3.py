from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)
    print("ID3 Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    return model
