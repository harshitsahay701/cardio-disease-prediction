Cardiovascular Disease Prediction

This project is about predicting whether a person might have cardiovascular disease using some medical data. We use machine learning models like Logistic Regression, Support Vector Machine (SVM), and Decision Trees to analyze the data and make predictions. The project also includes ways to check how well these models perform by looking at accuracy scores and different plots.

Project Structure
The project files are organized like this:
data/ — contains the dataset file named cardio_train.csv
models/ — contains the code files for different machine learning models
evaluation/ — contains code for plotting confusion matrices, ROC curves, and precision-recall graphs
utils/ — any helper scripts, like for data preprocessing
main.py — the main script that runs the whole project
requirements.txt — the list of Python libraries needed
README.md — this file explaining the project

Models Used
We used three main models for this project:
Logistic Regression — a simple linear model good for binary classification
Support Vector Machine (SVM) — a powerful method that finds the best boundary between classes
Decision Tree (ID3 algorithm) — a tree-based method that splits data based on information gain

Features
The data is preprocessed to standardize the values
We train each model on the data
Then we check how accurate the models are
We also generate plots like confusion matrices and ROC curves to visualize performance

Evaluation Metrics
We look at several things to see how well our models do:
Accuracy — percentage of correct predictions
Confusion Matrix — shows true vs predicted labels
ROC Curve — shows tradeoff between true positive and false positive rates
Precision-Recall Curve — useful when classes are imbalanced

How to Use This Project
First, clone this repository on your computer using this command:
git clone https://github.com/yourusername/cardio-disease-prediction.git
cd cardio-disease-prediction

Install the necessary Python libraries by running:
pip install -r requirements.txt

Download the dataset (cardio_train.csv) from Kaggle here:
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
Place this file inside the data folder.

Run the main program with:
python main.py

Required Libraries
You’ll need these Python packages:
pandas
numpy
scikit-learn
matplotlib
seaborn

About the Dataset
The dataset has many health-related features such as:
Age
Gender
Height and Weight
Blood pressure (systolic and diastolic)
Cholesterol and glucose levels
Habits like smoking, alcohol consumption, and physical activity
The target column, named cardio, tells if the person has cardiovascular disease (1) or not (0).

About Me
I am a beginner in machine learning and this project is my way of learning how to use Python and ML techniques on real health data.

License
This project is free to use for learning and educational purposes.
