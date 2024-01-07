from flask import Flask, request, jsonify 
import requests
import os 
import tempfile
from werkzeug.utils import secure_filename

import nltk
nltk.download("stopwords")
from pyresparser import ResumeParser
import PyPDF2

import json
import spacy
from spacy.matcher import Matcher


from flask import Flask, request, jsonify
from flask_cors import CORS


nlp = spacy.load('en_core_web_sm')

skill_matcher = Matcher(nlp.vocab)
college_matcher = Matcher(nlp.vocab)
softskill_matcher = Matcher(nlp.vocab)

#PDF TEXT PARSER
def pdf_to_text(file_path):
    # Open the PDF file
    with open(file_path, 'rb') as file:
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize text variable to store all the text
        text = ""

        # Loop through each page and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"

        return text

#PATTERN LOADER
def add_patterns_from_jsonl(file_path, matcher):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            label = data['label']
            pattern = data['pattern']
            matcher.add(label, [pattern])

#SKILL PROCESSOR
def preprocess_skill(text, file_path):
    add_patterns_from_jsonl(file_path, skill_matcher)
    doc = nlp(text)
    matches = skill_matcher(doc)

    extracted_skills = []
    for match_id, start, end in matches:
        # Extract only the skill part of the match
        skill = doc[start:end].text.lower()
        if skill not in list(map(str.lower,extracted_skills)):  # Check to avoid duplicates
            extracted_skills.append(doc[start:end].text)

    return extracted_skills

#COLLEGE_NAME PROCESSOR
def preprocess_college(text, file_path):
    add_patterns_from_jsonl(file_path, college_matcher)
    doc = nlp(text)
    matches = college_matcher(doc)

    extracted_college_name = []
    for match_id, start, end in matches:
        # Extract only the skill part of the match
        college_name = doc[start:end].text.lower()
        if college_name not in list(map(str.lower,extracted_college_name)):  # Check to avoid duplicates
            extracted_college_name.append(doc[start:end].text)

    return extracted_college_name

# SOFTSKILL PROCESSOR
def preprocess_softskill(text, file_path):
    add_patterns_from_jsonl(file_path, softskill_matcher)
    doc = nlp(text)
    matches = softskill_matcher(doc)

    extracted_softskill_name = []
    for match_id, start, end in matches:
        # Extract only the skill part of the match and convert to lowercase
        softskill_name = doc[start:end].text.lower()
        if softskill_name not in list(map(str.lower,extracted_softskill_name)):  # Check to avoid duplicates
            extracted_softskill_name.append(doc[start:end].text)

    return extracted_softskill_name


app = Flask(__name__)  
CORS(app)

@app.route('/process_resume', methods=['POST'])
def process_resume():
    # Extract URL from the request
    resume_url = request.json.get('resume_url')

    if not resume_url:
        return jsonify({"error": "No resume URL provided"}), 400

    # Download the PDF file
    try:
        response = requests.get(resume_url)
        response.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(response.content)
        file_path = temp_file.name

    # Process the file
    raw_text = pdf_to_text(file_path)
    processed_skills = preprocess_skill(raw_text, './skill_patterns.jsonl')
    processed_college = preprocess_college(raw_text, './university_patterns.jsonl')
    processed_softskill = preprocess_softskill(raw_text, './softskill_patterns.jsonl')

    data = ResumeParser(file_path).get_extracted_data()
    data['skills'] = processed_skills
    data['softskills'] = processed_softskill
    data['college_name'] = processed_college

    # Optionally remove 'experience' or other fields
    del data['experience']

    # Delete the temporary file
    os.remove(file_path)

    return jsonify(data)

@app.route('/process_job', methods=['POST'])
def process_job():
    job_desc = request.json.get('job_desc')
    processed_skills = preprocess_skill(job_desc, r'./skill_patterns.jsonl')
    processed_softskill = preprocess_softskill(job_desc, r'./softskill_patterns.jsonl')

    data = {}
    data['skills'] = processed_skills
    data['softskills'] = processed_softskill

    return jsonify(data)

@app.route('/', methods=['GET'])
def index():
    return 'v1.2'

if __name__ == '__main__':
    app.run(debug=True)
