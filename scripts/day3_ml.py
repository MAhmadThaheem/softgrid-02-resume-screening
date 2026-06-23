import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

os.makedirs('src/models', exist_ok=True)
os.makedirs('docs', exist_ok=True)

print("Loading feature engineered dataset...")
df = pd.read_csv('data/processed/feature_engineered_dataset.csv')

# The target is 'hired' (0 or 1)
y = df['hired']
X = df.drop(columns=['hired', 'candidate_id'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Naive Bayes': GaussianNB()
}

results = []

print("Training models...")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Save the model
    model_filename = f"src/models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, model_filename)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Confusion Matrix': str(cm.tolist())
    })

# Save feature columns to ensure consistency in predictions
joblib.dump(X.columns.tolist(), 'src/models/feature_columns.pkl')

results_df = pd.DataFrame(results)

# Create a Markdown Report
report_content = "# Model Comparison Report\n\n"
report_content += "## Overview\nThis report compares the performance of four classification models trained to predict candidate hiring suitability.\n\n"
report_content += "## Performance Metrics\n"

for i, row in results_df.iterrows():
    report_content += f"### {row['Model']}\n"
    report_content += f"- **Accuracy**: {row['Accuracy']:.4f}\n"
    report_content += f"- **Precision**: {row['Precision']:.4f}\n"
    report_content += f"- **Recall**: {row['Recall']:.4f}\n"
    report_content += f"- **F1 Score**: {row['F1 Score']:.4f}\n"
    report_content += f"- **Confusion Matrix**: `{row['Confusion Matrix']}`\n\n"

with open('docs/Model_Comparison_Report.md', 'w') as f:
    f.write(report_content)

# We will generate a Jupyter Notebook using jupytext
notebook_code = """# %% [markdown]
# # Day 3: Machine Learning Model Development
# 
# ## 1. Loading the Data
# %%
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('../data/processed/feature_engineered_dataset.csv')
y = df['hired']
X = df.drop(columns=['hired', 'candidate_id'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## 2. Training Models
# %%
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Naive Bayes': GaussianNB()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})
    print(f"{name} Accuracy: {acc:.4f}")
    
# %% [markdown]
# ## 3. Comparison
# %%
results_df = pd.DataFrame(results)
print(results_df)
"""

with open('notebooks/Day3_ML_Models.py', 'w') as f:
    f.write(notebook_code)

os.system('python -m jupytext --to notebook notebooks/Day3_ML_Models.py')
os.remove('notebooks/Day3_ML_Models.py')
print("Day 3 ML Models complete.")
