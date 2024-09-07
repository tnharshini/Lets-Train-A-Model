from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

app = Flask(__name__)

def load_or_initialize_model():
    if os.path.exists('sentiment_model.pkl') and os.path.exists('vectorizer.pkl'):
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        print("Loaded existing model and vectorizer.")
    else:
        df = pd.read_csv('sentiments.csv')
        X = df['text']
        y = df['label']
        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        save_model(model, vectorizer)
        print("Initialized new model and vectorizer.")
    return model, vectorizer

def save_model(model, vectorizer):
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

def update_and_retrain_model(user_input, new_label):
    new_data = pd.DataFrame({'text': [user_input], 'label': [new_label]})
    new_data.to_csv('sentiments.csv', mode='a', header=False, index=False)

    df = pd.read_csv('sentiments.csv')
    X = df['text']
    y = df['label']
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    save_model(model, vectorizer)
    return model, vectorizer

model, vectorizer = load_or_initialize_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    comment = data.get('comment')
    comment_vectorized = vectorizer.transform([comment])
    prediction = model.predict(comment_vectorized)[0]
    return jsonify({'prediction': prediction})

@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.json
    comment = data.get('comment')
    new_label = data.get('new_label')
    if new_label not in ['positive', 'negative', 'neutral']:
        return jsonify({'message': 'Invalid label'})
    model, vectorizer = update_and_retrain_model(comment, new_label)
    return jsonify({'message': 'Model updated'})

if __name__ == '__main__':
    app.run(debug=True)
