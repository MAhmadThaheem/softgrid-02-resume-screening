# Project Proposal: AI-Powered Resume Screening System

## 1. Problem Statement
In modern recruitment, HR departments receive hundreds or thousands of resumes for a single job opening. Manually screening each resume is a time-consuming, bias-prone, and inefficient process. Many qualified candidates might be overlooked due to human error or fatigue. There is a critical need for an automated Applicant Tracking System (ATS) that can intelligently parse, analyze, and categorize resumes based on machine learning algorithms.

## 2. Project Scope and Objectives
**Objective:** To develop a machine learning-based web application that automatically classifies resumes into predefined job categories and predicts the candidate's suitability for a role.

**Scope:**
- **Data Collection & Preprocessing:** Acquire a text-based resume dataset and clean the data (remove stop words, punctuation, apply stemming/lemmatization).
- **Feature Engineering:** Convert text into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency) or Word Embeddings.
- **Model Training:** Train multiple classification models (Logistic Regression, Random Forest, Naive Bayes, etc.) to categorize resumes.
- **Web Interface:** Develop a Flask-based web application where recruiters can upload resume files (PDF/DOCX) and get instant categorization and suitability scores.
- **Evaluation:** Compare model accuracies to select the best-performing model for the production deployment.

## 3. Input Features and Expected Outputs
**Input Features:**
- **Resume Text:** The raw text extracted from uploaded resume files (PDF, DOCX, or TXT). This includes education, work experience, skills, and summary sections.
- **Target Role (Optional):** The job role the candidate is applying for (to compute a match/confidence score).

**Expected Outputs:**
- **Predicted Job Category:** The domain or industry the resume belongs to (e.g., Data Science, Web Development, HR, Sales).
- **Matching Skills:** Key skills extracted from the resume.
- **Confidence Score:** The model's probability score indicating how confident it is about the predicted category.

## 4. Dataset Description
The system will utilize a structured dataset containing labeled resumes. 
- **Source:** Kaggle (e.g., "Updated Resume Dataset").
- **Features:**
  - `Category`: The target label (e.g., 'Data Science', 'Java Developer').
  - `Resume`: The text content of the resume.
- **Size:** Typically 500 - 1000+ records to ensure the machine learning models have sufficient data to learn patterns.
