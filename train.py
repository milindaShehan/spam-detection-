from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Assuming the dataset has 'Category' and 'Message' columns
    X = data['Message']
    y = data['Category'].apply(lambda x: 1 if x == 'ham' else 0)  # Convert 'ham' to 1 and 'spam' to 0
    return X, y

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', binary=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_features, y_train)
    
    return model, feature_extraction

# Save the model and feature extraction tools
def save_model(model, feature_extraction):
    pickle.dump(model, open('logistic_regression.pkl', 'wb'))
    pickle.dump(feature_extraction, open('feature_extraction.pkl', 'wb'))

if __name__ == "__main__":
    dataset_path = 'mail_data.csv'  # Update with your dataset path
    data = load_data(dataset_path)
    X, y = preprocess_data(data)
    model, feature_extraction = train_model(X, y)
    save_model(model, feature_extraction)
    print("Model trained and saved successfully.")