# Sentiment_Analysis_Pipeline

🚀 **An End-to-End Sentiment Analysis Pipeline** that classifies IMDB movie reviews as **positive** or **negative** using **Logistic Regression & TF-IDF**.

---

## **Project Overview**
This project implements a **sentiment analysis pipeline**:
✅ Loads the **IMDB movie reviews dataset**.  
✅ Stores the data in a **SQLite database**.  
✅ Trains a **Logistic Regression model with TF-IDF vectorization**.  
✅ Exposes the trained model through a **Flask API** for predictions.  

---

## **Project Setup**
### **Install Dependencies **
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/Sentiment_Analysis.git


### Create a virtual environment(optional but recommended):
```bash
    python -m venv sentiment_env
```

*  **Activate Virtual Environment:**
    * Windows: sentiment_env\Scripts\activate
    * Mac/Linux: source sentiment_env/bin/activate
*  **Install dependencies** using requirements.txt:
  ```bash
    pip install -r requirements.txt
```

##  **Data Acquisition**
  **How the Dataset Was Obtained**
 * The dataset is the IMDB Movie Reviews Dataset, containing 50,000 labeled movie reviews.
 * We used the Hugging Face Datasets Library to load it directly into a Pandas DataFrame.
``` bash
  from datasets import load_dataset
dataset = load_dataset("imdb")
```
* The dataset includes:
    * **Train Set**: 25,000 reviews.
    * **Test Set**: 25,000 reviews.
    * **Labels**: "positive" (1) or "negative" (0).
 
## **Data Collection & Database Setup**
  **Jupyter Notebook: DataCollection_And_DatabaseSetup.ipynb**
This notebook: ✅ Loads the IMDB dataset using Hugging Face.
✅ Stores the dataset in a SQLite database (imdb_reviews.db).
✅ Creates a table (imdb_reviews) with:
* id (Primary Key)
* review_text (Review content)
* sentiment (positive/negative)
* data_split (train/test)
  **To run the notebook:**
  ```bash
  jupyter notebook DataCollection_And_DatabaseSetup.ipynb
  ```
This step ensures that all data is properly stored before training.

## **Data Cleaning** & **Exploratory Data Analysis(EDA)**
 **Jupyter Notebook: DataCleaning_And_Exploration.ipynb**
 This notebook: ✅ Removes duplicates & missing values
✅ Performs text preprocessing (lowercasing, removing HTML tags, punctuation)
✅ EDA:

* Class distribution (positive vs. negative reviews)
* Word frequency plots
* Average review length analysis
  **To run the notebook:**
  ```bash
      jupyter notebook DataCleaning_And_Exploration.ipynb
  ```

  This step ensures clean input data for training.

  ## **Model Training & Evaluation**
 **Jupyter Notebook: ModelTraining_And_Evaluation.ipynb**
 This notebook: ✅ Loads cleaned training data from SQLite
✅ Applies TF-IDF vectorization
✅ Trains a Logistic Regression model
✅ Saves:
* Trained model (models/logistic_regression_model.pkl)
* TF-IDF vectorizer (models/tfidf_vectorizer.pkl) ✅ Evaluates the model:
* Accuracy, Precision, Recall, F1-score

  **To run the notebook:**
  ```bash
     jupyter notebook ModelTraining_And_Evaluation.ipynb
  ```

 ## **Running the Flask API**
 **File: app.py**
 This script: ✅ Loads the trained Logistic Regression model.
✅ Starts a Flask API for real-time predictions.
Run the API locally:
```bash
python app.py
```
Expected Output:
```bash
 Running on http://127.0.0.1:5000/
```

## **Testing the API**
 ### Using Postman
1. Open **Postman**.
2. Select **POST**.
3. Enter the **URL**:
   
   ```
   http://127.0.0.1:5000/predict
   ```

4. Go to the **Body** tab → Select **raw** → Choose **JSON**.
5. Enter the following JSON
```
{
  "review_text": "I absolutely loved this movie. It was fantastic!"
}
  ```

6. Click Send.

**Expected Response:**

   ```
{
  "sentiment_prediction": "positive"
}

   ```
## **Model Information**
**Model Approach**
The model is a Logistic Regression classifier trained on TF-IDF vectorized text data.

| Feature              | Value                              |
|----------------------|----------------------------------|
| **Model Type**       | Logistic Regression             |
| **Feature Extraction** | TF-IDF (Max Features = 10,000) |
| **Training Accuracy** | ~89%                            |
| **Test Accuracy**    | ~88.5%                           |

## **Deployment on Render**
**Steps to Deploy on Render**
1️⃣ Ensure all required files are pushed to GitHub:

* app.py (Flask API)
* requirements.txt
* runtime.txt (Optional: To set Python version)
* models/ (Trained models, e.g., logistic_regression_model.pkl, tfidf_vectorizer.pkl)
2️⃣ Create a Free Account on Render
  * Sign up at Render.
3️⃣ Deploy a New Web Service
* Click "New Web Service" → Connect GitHub Repo.
* Select your repository (Sentiment_Analysis).
* Set the following:
   * Build Command:
    ```
    pip install -r requirements.txt
   ```
  * Start Command:
  ```
     gunicorn app:app
  ```
  
* Click Deploy

4️⃣ Wait for Deployment to Complete.

* Once deployed, Render provides a public API URL.

## **Testing the API After Deployment**
 Once deployed, test the API using:
✅ Postman
1️⃣ Open Postman
2️⃣ Create a New Request:

* Method: POST
* URL
```
https://your-api-name.onrender.com/predict

```

3️⃣ Go to the Headers Tab:
* Add:
```
Key: Content-Type
Value: application/json
```
4️⃣ Go to the Body Tab:

* Select raw → Choose JSON.
* Enter:
```
{
  "review_text": "I absolutely loved this movie. It was fantastic!"
}

```
5️⃣ Click "Send"
✅ Expected Response:
```
{
    "sentiment_prediction": "positive"
}

```
