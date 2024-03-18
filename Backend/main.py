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
# import time
# import requests.exceptions
from transformers import pipeline
import re

app = Flask(__name__)

# Configure your OpenAI API key
openai.api_key = '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'

# Load the plagiarism detection model
vectorizer = TfidfVectorizer()
classifier = joblib.load('plagiarism_detection_model.pkl')
training_data = [
    "Researchers have discovered a new species of butterfly in the Amazon rainforest.",
    "Scientists have found a previously unknown butterfly species in the Amazon jungle.",
    "The moon orbits the Earth in approximately 27.3 days.",
    "Our natural satellite takes around 27.3 days to complete one orbit around our planet.",
    "Water is composed of two hydrogen atoms and one oxygen atom.",
    "H2O consists of 2 hydrogen atoms and 1 oxygen atom.",
    "The history of Rome dates back to 753 BC.",
    "Rome has a long history that can be traced back to 753 BC.",
    "Pluto was once considered the ninth planet in our solar system.",
    "In the past, Pluto was classified as the ninth planet in our sun's planetary system.",
    "This is a unique and original sentence.",
    "This sentence is unique and original.",
    "Artificial intelligence is reshaping industries.",
    "AI is changing the landscape of various sectors.",
    "Python is a popular programming language for data science.",
    "Data science often relies on Python as a widely used programming language.",
    "The Earth revolves around the Sun in a nearly circular orbit.",
    "Our planet follows an almost circular path as it moves around the central star.",
    "Paris is the capital of France.",
    "France's capital city is Paris."
]

vectorizer.fit(training_data)

# def is_ai_generated(text):
#     retries = 3  # Number of retries
#     delay = 1  # Initial delay in seconds

#     for _ in range(retries):
#         try:
#             response = openai.Completion.create(
#                 engine="davinci-002",
#                 prompt=text,
#                 max_tokens=50,
#                 temperature=0.7
#             )
#             generated_text = response.choices[0].text.strip()
#             # Calculate the percentage of text generated using AI
#             percentage_generated = len(generated_text) / len(text) * 100
#             return percentage_generated
#         except openai.error.RateLimitError as e:
#             print("Rate limit exceeded. Waiting and retrying...")
#             time.sleep(delay)  # Wait before retrying
#             delay *= 2  # Exponential backoff for delay
#         except requests.exceptions.RequestException as e:
#             print("Request failed:", e)
#             break  # Exit loop if other types of exceptions occur

#     return 0  # Return 0 if unable to generate text or encounter an error
def calculate_similarity_percentage(text1, text2):
    # Tokenize the text by whitespace
    tokens1 = re.findall(r"\w+", text1.lower())
    tokens2 = re.findall(r"\w+", text2.lower())
    
    # Calculate the number of common tokens
    common_tokens = len(set(tokens1) & set(tokens2))
    
    # Calculate the percentage of matching tokens
    similarity_percentage = (common_tokens / len(tokens2)) * 100
    return similarity_percentage

def is_ai_generated(text):
    text_generation_pipeline = pipeline("text-generation", model="gpt2")
    generated_text = text_generation_pipeline(text, max_length=50, num_return_sequences=1)[0]['generated_text'].strip()
    similarity_percentage = calculate_similarity_percentage(text, generated_text)
    return similarity_percentage


# Function to check plagiarism
def check_plagiarism(text1, text_list):
    total_similarity = 0
    for text2 in text_list:
        vectors = vectorizer.transform([text1, text2])
        similarity = cosine_similarity(vectors)
        total_similarity += similarity[0, 1]
    avg_similarity = total_similarity / len(text_list)
    plagiarism_percentage = avg_similarity * 100
    return plagiarism_percentage


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

        # Split report_text into sentences
        report_sentences = [sentence.strip() for sentence in report_text.split('.') if sentence.strip()]

        # Save extracted text to CSV file
        csv_file = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_text.csv')
        save_to_csv(report_text, csv_file)

       
        ai_generated = is_ai_generated(report_text)
        plagiarism_percentage = check_plagiarism(report_text, report_sentences)

        return jsonify({
            "ai_generated": ai_generated,
            "plagiarism_percentage": plagiarism_percentage
        })



if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
