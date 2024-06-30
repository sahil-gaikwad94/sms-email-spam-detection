import pandas as pd

# Load the CSV file to inspect its content
file_path = 'spam.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Display the first few rows of the dataframe to understand its structure
data.head()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Clean the data by removing unnecessary columns and renaming the important ones
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to binary values: 'ham' -> 0, 'spam' -> 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

X_train_tfidf.shape, X_test_tfidf.shape


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predict the labels on the test set
y_pred_nb = nb_classifier.predict(X_test_tfidf)

# Evaluate the model
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)
nb_classification_report = classification_report(y_test, y_pred_nb)

nb_accuracy, nb_precision, nb_recall, nb_f1, nb_classification_report


from sklearn.linear_model import LogisticRegression

# Train a Logistic Regression classifier
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train_tfidf, y_train)

# Predict the labels on the test set
y_pred_lr = lr_classifier.predict(X_test_tfidf)

# Evaluate the model
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_classification_report = classification_report(y_test, y_pred_lr)

lr_accuracy, lr_precision, lr_recall, lr_f1, lr_classification_report

import joblib
joblib.dump(nb_classifier, 'model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')