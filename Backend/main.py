from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import docx
import os
import joblib
import csv

app = Flask(__name__)

# Configure your OpenAI API key
openai.api_key = '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'

# Load the plagiarism detection model
vectorizer = TfidfVectorizer()
classifier = joblib.load('plagiarism_detection_model.pkl')

# Function to check if text is AI-generated
def is_ai_generated(text):
    response = openai.Completion.create(
        engine="distilgpt2",
        prompt=text,
        max_tokens=50,
        temperature=0.7
    )
    generated_text = response.choices[0].text.strip()
    return generated_text.lower() == text.lower()

# Function to check plagiarism
def check_plagiarism(text1, text2):
    vectors = vectorizer.transform([text1, text2])
    similarity = cosine_similarity(vectors)
    return similarity[0, 1]

# Function to extract text from Word file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Function to save text to CSV file
def save_to_csv(text, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["source_text"])
        writer.writerow([text])

# Endpoint to handle Word file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from Word file
        report_text = extract_text_from_docx(file_path)

        # Save extracted text to CSV file
        csv_file = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_text.csv')
        save_to_csv(report_text, csv_file)

        # Check for plagiarism
        ai_generated = is_ai_generated(report_text)
        plagiarism_score = check_plagiarism(report_text, "Reference text")

        return jsonify({
            "ai_generated": ai_generated,
            "plagiarism_score": plagiarism_score
        })

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
