from flask import Flask, request, jsonify
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Step 1: Load the model and TF-IDF vectorizer
print("Loading model and TF-IDF vectorizer...")
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

print("Model and vectorizer loaded successfully!")

# Step 2: Define text preprocessing function
def preprocess_text(text):
    """
    Preprocess the input text to match the training process:
    - Lowercase conversion
    - Remove HTML tags
    - Remove punctuation
    """
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

# Step 3: Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict the sentiment of a given review.
    """
    # Parse JSON input
    input_data = request.get_json()
    if "review_text" not in input_data:
        return jsonify({"error": "Missing 'review_text' field in request"}), 400

    # Extract and preprocess the input text
    review_text = input_data["review_text"]
    preprocessed_text = preprocess_text(review_text)

    # Transform text using the TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])

    # Predict sentiment using the trained model
    prediction = model.predict(vectorized_text)[0]
    sentiment = "positive" if prediction == 1 else "negative"

    # Return the result as JSON
    return jsonify({"sentiment_prediction": sentiment})

# Step 4: Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
