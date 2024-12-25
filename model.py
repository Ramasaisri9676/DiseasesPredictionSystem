import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import joblib
import re
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('disease_dataset.csv')
df.columns = df.columns.str.strip()  # Strip whitespace from column names

# Handle missing values
df = df.dropna()  # Drop rows with missing values

# Text preprocessing for symptoms
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing whitespace
    return text

df['symptoms'] = df['symptoms'].apply(preprocess_text)

# Encode target variables (if necessary)
label_encoders = {}  # Dictionary to store label encoders for each target variable
for col in ['disease', 'cures', 'doctor', 'risk level']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target variables
X = df['symptoms']
y = df[['disease', 'cures', 'doctor', 'risk level']]

# Text vectorization
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Model training
model = MultinomialNB()
multi_output_model = MultiOutputClassifier(model, n_jobs=-1)
multi_output_model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(multi_output_model, 'multi_output_disease_predict_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Save label encoders (if used)
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model, vectorizer, and label encoders saved.")
