# AI-Powered Resume Screening System

## 📌 Project Overview
This project is an AI-powered Resume Screening System designed to automatically analyze resumes, extract relevant features, and predict a candidate's suitability for a job role. It reduces manual effort in the hiring process, minimizes bias, and accelerates recruitment. The system was built as part of the SoftGrid Solutions 5-day internship.

## 📂 Dataset Information
- **Source**: Kaggle / Synthetic Tabular Dataset
- **Size**: 200,000+ records
- **Features Include**: `age`, `education_level`, `university_tier`, `cgpa`, `internships`, `projects`, `programming_languages`, `certifications`, `experience_years`, `skills_score`, `soft_skills_score`, `resume_length_words`, `company_type`.
- **Target Variable**: `hired` (1 for highly suitable, 0 for not suitable).

## 🛠️ Technologies Used
- **Language**: Python 3.x
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Logistic Regression, Random Forest, Decision Tree, Naive Bayes)
- **Web Framework**: Flask
- **Text Extraction**: pdfplumber, python-docx
- **Frontend**: HTML5, CSS3

## 🚀 Installation Guide
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd softgrid-02-resume-screening
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Data Preprocessing (Day 2):
   ```bash
   python scripts/day2_eda.py
   ```
4. Train the ML Models (Day 3):
   ```bash
   python scripts/day3_ml.py
   ```

## 💻 Usage Instructions
1. Start the Flask application:
   ```bash
   cd web_app
   python app.py
   ```
2. Open your web browser and navigate to `http://127.0.0.1:5000/`.
3. Upload a Resume in `.pdf` or `.docx` format.
4. The system will automatically parse the resume, extract relevant features, pass them to the trained Random Forest model, and display a "Highly Suitable" or "Not Suitable" result along with a confidence score.

## 📸 Screenshots
*(Add screenshots of your web app running, the upload page, and the prediction result page here)*

## 🔮 Future Improvements
- Implement advanced NLP (like SpaCy or HuggingFace Transformers) to better extract structured data from raw resume text rather than using heuristics.
- Train the model to predict the *specific Job Category* instead of just a binary hiring outcome.
- Deploy the application to a cloud platform like AWS or Heroku.
