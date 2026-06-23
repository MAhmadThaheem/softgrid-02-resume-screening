import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

os.makedirs('docs/model_visualizations', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('data/processed/feature_engineered_dataset.csv')
y = df['hired']
X = df.drop(columns=['hired', 'candidate_id'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models_to_evaluate = {
    'Logistic Regression': 'src/models/logistic_regression.pkl',
    'Random Forest': 'src/models/random_forest.pkl',
    'Decision Tree': 'src/models/decision_tree.pkl',
    'Naive Bayes': 'src/models/naive_bayes.pkl'
}

metrics_data = []

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

print("Evaluating models and generating visuals...")
for idx, (name, path) in enumerate(models_to_evaluate.items()):
    if os.path.exists(path):
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics_data.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        # Plot Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    else:
        print(f"Model {name} not found at {path}")

plt.tight_layout()
plt.savefig('docs/model_visualizations/confusion_matrices.png')
plt.close()

# Plot Metrics Comparison
metrics_df = pd.DataFrame(metrics_data)
metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette='viridis')
plt.title('Model Performance Comparison (Different Perspectives)')
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('docs/model_visualizations/metrics_comparison.png')
plt.close()

print("Visualizations successfully generated and saved to docs/model_visualizations/")
