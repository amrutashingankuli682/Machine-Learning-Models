import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
# Load the Dataset
df = pd.read_csv('diabetes.csv')
feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']
target_names=['Outcome']

#print(df.head())
# Split the Data
X = df[feature_names].values
y = df[target_names].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Train the Model
model = GaussianNB()
model.fit(X_train, y_train.ravel())
# Evaluate the Model
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))