from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

docs = [
    "We have a good team working on project",

    "employees are not happy",
    "Buisness have a huge loss",
    "Clients are unhappy",
    "Good employees work",
    "flexible working hours",
    "good salary for employees",
    "few projects have failed",
]
labels=['positive','negative','negative','negative','positive','positive','positive','negative']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(labels)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print(f'Classification Report:\n{classification_report(Y_test, Y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}')

# new_text = "Employees are working hard."
# new_text_transformed = vectorizer.transform([new_text])
# ans = clf.predict(new_text_transformed)
# print(f'Prediction for the new text: {label_encoder.classes_[ans[0]]}')