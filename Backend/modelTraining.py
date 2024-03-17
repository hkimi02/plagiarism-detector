import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def load_data_from_csv(csv_file):
    documents = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            document = {
                "text": row['source_text'],
                "plagiarized": int(row['label'])  # Convert string label to integer
            }
            documents.append(document)
    return documents

# Load data from CSV file
csv_file = 'your_data.csv'  # Replace 'your_csv_file.csv' with the path to your CSV file
documents = load_data_from_csv(csv_file)

# Preprocess and extract features
texts = [doc["text"] for doc in documents]
labels = [doc["plagiarized"] for doc in documents]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save the model to disk
joblib.dump(classifier, 'plagiarism_detection_model.pkl')
