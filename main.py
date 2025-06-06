from utils.preprocessing import load_and_preprocess_data
from models.linear_regression import train_linear_regression
from models.svm_model import train_svm_model
from models.decision_tree_id3 import train_decision_tree
from evaluation.confusion_matrix_plot import plot_confusion_matrix
from evaluation.roc_curve_plot import plot_roc_curve
from evaluation.precision_recall_plot import plot_precision_recall

X_train, X_test, y_train, y_test = load_and_preprocess_data('data/cardio_train.csv')

lr_model = train_linear_regression(X_train, y_train, X_test, y_test)
svm_model = train_svm_model(X_train, y_train, X_test, y_test)
id3_model = train_decision_tree(X_train, y_train, X_test, y_test)

plot_confusion_matrix(lr_model, X_test, y_test, "Linear Regression")
plot_roc_curve(svm_model, X_test, y_test, "SVM")
plot_precision_recall(id3_model, X_test, y_test, "ID3")
