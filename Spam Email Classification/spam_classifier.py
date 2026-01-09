import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class SpamClassifier:
    def __init__(self, model_type='nb'):
        """
        Initialize Spam Classifier

        Parameters:
        model_type: 'nb' for Naive Bayes, 'lr' for Logistic Regression,
                   'svm' for Support Vector Machine, 'rf' for Random Forest
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None

    def preprocess_text(self, text):
        """
        Preprocess text: clean, tokenize, lemmatize, remove stopwords
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Lemmatize and remove stopwords
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens
                  if word not in self.stop_words and len(word) > 2]

        return ' '.join(tokens)

    def preprocess_data(self, data):
        """
        Preprocess entire dataset
        """
        print("Preprocessing text data...")
        processed_texts = data['text'].apply(self.preprocess_text)
        return processed_texts

    def train(self, X_train, y_train):
        """
        Train the model
        """
        print("Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)

        print(f"Training {self.model_type.upper()} model...")

        if self.model_type == 'nb':
            self.model = MultinomialNB()
        elif self.model_type == 'lr':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True, random_state=42)
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Invalid model type. Choose from 'nb', 'lr', 'svm', 'rf'")

        self.model.fit(X_train_vec, y_train)
        print(f"{self.model_type.upper()} model training completed!")

        return X_train_vec

    def predict(self, X):
        """
        Make predictions
        """
        X_vec = self.vectorizer.transform(X)
        predictions = self.model.predict(X_vec)
        probabilities = self.model.predict_proba(X_vec)
        return predictions, probabilities

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        predictions, probabilities = self.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Not Spam', 'Spam'])
        conf_matrix = confusion_matrix(y_test, predictions)

        print(f"\nModel Evaluation - {self.model_type.upper()}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'probabilities': probabilities
        }

    def save_model(self, filename_prefix='spam_classifier'):
        """
        Save the trained model and vectorizer to disk
        """
        model_filename = f"{filename_prefix}_{self.model_type}.pkl"
        vectorizer_filename = f"{filename_prefix}_vectorizer.pkl"

        with open(model_filename, 'wb') as f:
            pickle.dump(self.model, f)

        with open(vectorizer_filename, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        print(f"Model saved as: {model_filename}")
        print(f"Vectorizer saved as: {vectorizer_filename}")

        return model_filename, vectorizer_filename

    def load_model(self, model_filename, vectorizer_filename):
        """
        Load a trained model and vectorizer from disk
        """
        with open(model_filename, 'rb') as f:
            self.model = pickle.load(f)

        with open(vectorizer_filename, 'rb') as f:
            self.vectorizer = pickle.load(f)

        print(f"Model loaded from: {model_filename}")
        print(f"Vectorizer loaded from: {vectorizer_filename}")


def load_and_prepare_data(csv_path):
    """
    Load and prepare the dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Spam distribution:\n{df['spam'].value_counts()}")

    return df


def train_and_evaluate_models(df, save_models=True):
    """
    Train and evaluate multiple models
    """
    # Preprocess the text data
    classifier = SpamClassifier()
    processed_texts = classifier.preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, df['spam'], test_size=0.2, random_state=42, stratify=df['spam']
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train and evaluate different models
    models_results = {}
    models_to_train = ['nb', 'lr', 'svm', 'rf']

    for model_type in models_to_train:
        print(f"\n{'=' * 50}")
        print(f"Training {model_type.upper()} model")
        print('=' * 50)

        # Create and train classifier
        classifier = SpamClassifier(model_type=model_type)

        # Train model
        classifier.train(X_train, y_train)

        # Evaluate model
        results = classifier.evaluate(X_test, y_test)
        models_results[model_type] = results

        # Save model if requested
        if save_models:
            classifier.save_model()

    return models_results, X_test, y_test


def analyze_model_performance(models_results):
    """
    Analyze and compare performance of all trained models
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    comparison_data = []
    for model_name, results in models_results.items():
        comparison_data.append({
            'Model': model_name.upper(),
            'Accuracy': results['accuracy'],
            'Precision (Spam)': classification_report(
                y_test,
                results['predictions'],
                output_dict=True
            )['1']['precision'],
            'Recall (Spam)': classification_report(
                y_test,
                results['predictions'],
                output_dict=True
            )['1']['recall'],
            'F1-Score (Spam)': classification_report(
                y_test,
                results['predictions'],
                output_dict=True
            )['1']['f1-score']
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))

    # Find best model
    best_model = max(models_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model[0].upper()} with accuracy: {best_model[1]['accuracy']:.4f}")

    return comparison_df


def predict_new_email(email_text, model_type='nb'):
    """
    Predict whether a new email is spam
    """
    # Load the saved model
    model_filename = f"spam_classifier_{model_type}.pkl"
    vectorizer_filename = "spam_classifier_vectorizer.pkl"

    try:
        classifier = SpamClassifier(model_type=model_type)
        classifier.load_model(model_filename, vectorizer_filename)

        # Preprocess the email text
        processed_text = classifier.preprocess_text(email_text)

        # Make prediction
        prediction, probabilities = classifier.predict([processed_text])

        is_spam = bool(prediction[0])
        spam_probability = probabilities[0][1]

        print(f"\nEmail Analysis:")
        print(f"Is Spam: {'YES' if is_spam else 'NO'}")
        print(f"Spam Probability: {spam_probability:.2%}")
        print(f"Ham Probability: {probabilities[0][0]:.2%}")

        return {
            'is_spam': is_spam,
            'spam_probability': spam_probability,
            'ham_probability': probabilities[0][0]
        }

    except FileNotFoundError:
        print(f"Model files not found. Please train the model first.")
        return None


def interactive_prediction():
    """
    Interactive mode for testing emails
    """
    print("\n" + "=" * 50)
    print("SPAM CLASSIFIER INTERACTIVE MODE")
    print("=" * 50)
    print("Enter email text (press Enter twice to submit):")

    email_lines = []
    while True:
        line = input()
        if line == "":
            if len(email_lines) > 0:
                break
        email_lines.append(line)

    email_text = "\n".join(email_lines)

    model_choice = input("\nChoose model (nb/lr/svm/rf) [default=nb]: ").lower() or 'nb'

    result = predict_new_email(email_text, model_choice)

    if result:
        print("\n" + "-" * 30)
        if result['is_spam']:
            print("⚠️  WARNING: This email is classified as SPAM")
        else:
            print("✅ This email is classified as HAM (Not Spam)")


if __name__ == "__main__":
    # Path to your CSV file
    CSV_PATH = "emails.csv"  # Update this path if needed

    try:
        # Step 1: Load data
        df = load_and_prepare_data(CSV_PATH)

        # Step 2: Train and evaluate models
        models_results, X_test, y_test = train_and_evaluate_models(df, save_models=True)

        # Step 3: Analyze performance
        comparison_df = analyze_model_performance(models_results)

        # Step 4: Save comparison results
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("\nModel comparison saved to 'model_comparison.csv'")

        # Step 5: Interactive prediction mode
        while True:
            print("\n" + "=" * 50)
            choice = input("\nDo you want to test a new email? (y/n): ").lower()

            if choice == 'y':
                interactive_prediction()
            else:
                print("\nExiting program. Models are saved in the current directory.")
                break

    except FileNotFoundError:
        print(f"Error: File '{CSV_PATH}' not found.")
        print("Please ensure the CSV file is in the correct path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")