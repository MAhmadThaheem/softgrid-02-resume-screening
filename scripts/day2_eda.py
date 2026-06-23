import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('data/processed', exist_ok=True)
os.makedirs('docs/eda_visualizations', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('data/resume_dataset_200k.csv')

print("Initial shape:", df.shape)

# 1. Remove duplicate records
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# 2. Handle missing values
# For numerical columns, fill with median. For categorical, fill with mode.
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled.")

# Save cleaned dataset
df.to_csv('data/processed/cleaned_dataset.csv', index=False)
print("Cleaned dataset saved.")

# 3. Exploratory Data Analysis & Visualizations
sns.set_theme(style="whitegrid")

# A. Category Distribution (education_level)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='education_level', palette='viridis')
plt.title('Education Level Distribution')
plt.tight_layout()
plt.savefig('docs/eda_visualizations/education_distribution.png')
plt.close()

# B. Target Distribution (hired)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='hired', palette='Set2')
plt.title('Hired (Target) Distribution')
plt.tight_layout()
plt.savefig('docs/eda_visualizations/hired_distribution.png')
plt.close()

# C. Experience vs Hired
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='hired', y='experience_years', palette='Set1')
plt.title('Experience Years vs Hiring Outcome')
plt.tight_layout()
plt.savefig('docs/eda_visualizations/experience_vs_hired.png')
plt.close()

# D. Skills Score vs Hired
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='hired', y='skills_score', palette='Set3')
plt.title('Skills Score vs Hiring Outcome')
plt.tight_layout()
plt.savefig('docs/eda_visualizations/skills_vs_hired.png')
plt.close()

print("EDA visualizations saved.")

# 4. Feature Engineering
# One-Hot Encoding categorical columns
cat_cols = ['education_level', 'university_tier', 'company_type']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Save feature-engineered dataset
df_encoded.to_csv('data/processed/feature_engineered_dataset.csv', index=False)
print("Feature engineered dataset saved.")

# We will generate a Jupyter Notebook using jupytext
notebook_code = """# %% [markdown]
# # Day 2: Data Preprocessing & Exploratory Data Analysis
# 
# ## 1. Loading the Data
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/resume_dataset_200k.csv')
print(df.head())

# %% [markdown]
# ## 2. Data Cleaning
# %%
df = df.drop_duplicates()
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

df.to_csv('../data/processed/cleaned_dataset.csv', index=False)

# %% [markdown]
# ## 3. EDA
# %%
sns.countplot(data=df, x='education_level')
plt.title('Education Level Distribution')
plt.show()

# %% [markdown]
# ## 4. Feature Engineering
# %%
cat_cols = ['education_level', 'university_tier', 'company_type']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df_encoded.to_csv('../data/processed/feature_engineered_dataset.csv', index=False)
print("Processing Complete")
"""

with open('notebooks/Day2_Preprocessing_EDA.py', 'w') as f:
    f.write(notebook_code)

os.system('python -m jupytext --to notebook notebooks/Day2_Preprocessing_EDA.py')
os.remove('notebooks/Day2_Preprocessing_EDA.py')
print("Notebook generated.")
