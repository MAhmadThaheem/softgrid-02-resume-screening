import os
import joblib
import pdfplumber
import docx
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model and feature columns
model_path = '../src/models/random_forest.pkl'
features_path = '../src/models/feature_columns.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
else:
    model = None
    feature_columns = None

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_features(text):
    # A simple heuristic extractor to map raw text to our tabular feature format.
    # In a real-world scenario, you would use NLP (SpaCy/Transformers) here.
    text_lower = text.lower()
    
    # Heuristics
    resume_length_words = len(text.split())
    age = 25 # default assumption
    cgpa = 7.5 # default assumption
    
    # Education
    edu_level = "Bachelors"
    if "master" in text_lower or "msc" in text_lower or "mba" in text_lower:
        edu_level = "Masters"
    elif "phd" in text_lower or "doctorate" in text_lower:
        edu_level = "PhD"
        
    university_tier = "Tier 2"
    if "iit" in text_lower or "nit" in text_lower or "stanford" in text_lower or "mit" in text_lower:
        university_tier = "Tier 1"
        
    # Counts
    programming_languages = sum(1 for word in ["python", "java", "c++", "javascript", "ruby", "go", "php"] if word in text_lower)
    skills_score = sum(1 for word in ["machine learning", "data science", "web", "react", "sql", "aws", "docker", "agile", "communication", "leadership"] if word in text_lower) * 2.5
    soft_skills_score = sum(1 for word in ["team", "lead", "manage", "communicate", "organize"] if word in text_lower) * 2.0
    experience_years = text_lower.count('experience') * 0.5
    internships = text_lower.count('intern')
    projects = text_lower.count('project')
    certifications = text_lower.count('certifi')
    hackathons = text_lower.count('hackathon')
    research_papers = text_lower.count('research') + text_lower.count('paper')
    company_type = "MNC" if "global" in text_lower or "international" in text_lower else "Startup"

    # Build row
    row = {
        'age': age,
        'cgpa': cgpa,
        'internships': internships,
        'projects': projects,
        'programming_languages': programming_languages,
        'certifications': certifications,
        'experience_years': experience_years,
        'hackathons': hackathons,
        'research_papers': research_papers,
        'skills_score': skills_score,
        'soft_skills_score': soft_skills_score,
        'resume_length_words': resume_length_words,
        'education_level': edu_level,
        'university_tier': university_tier,
        'company_type': company_type
    }
    
    return row

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return render_template('index.html', error="No file uploaded.")
        
        file = request.files['resume']
        if file.filename == '':
            return render_template('index.html', error="No selected file.")
            
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Extract text
            ext = os.path.splitext(file.filename)[1].lower()
            text = ""
            if ext == '.pdf':
                text = extract_text_from_pdf(filepath)
            elif ext == '.docx':
                text = extract_text_from_docx(filepath)
            else:
                return render_template('index.html', error="Unsupported file format. Please upload PDF or DOCX.")
            
            # Extract features
            extracted_features = extract_features(text)
            df_features = pd.DataFrame([extracted_features])
            
            # One-Hot Encode dynamically based on what model expects
            if feature_columns is not None:
                # One-hot encode what we have
                df_encoded = pd.get_dummies(df_features, columns=['education_level', 'university_tier', 'company_type'])
                # Realign columns with training data (fill missing with 0)
                for col in feature_columns:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                df_encoded = df_encoded[feature_columns]
                
                # Predict
                prediction = model.predict(df_encoded)[0]
                probability = model.predict_proba(df_encoded)[0].max() * 100
                
                result = "Highly Suitable / Hired" if prediction == 1 else "Not Suitable / Rejected"
                
                return render_template('result.html', 
                                       result=result, 
                                       confidence=round(probability, 2),
                                       features=extracted_features,
                                       summary=text[:300] + "...")
            else:
                return render_template('index.html', error="Model not trained yet. Please run Day 3 script.")
                
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
