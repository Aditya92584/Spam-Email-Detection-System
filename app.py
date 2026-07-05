from flask import Flask, request, jsonify, render_template
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK resources ko initialize karein
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# Model aur Vectorizer load karein
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
ps = PorterStemmer()

# Wahi Text Preprocessing Function jo Jupyter me banaya tha
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('email', '')

    if not email_text:
        return jsonify({'error': 'No text provided'}), 400

    # 1. Text preprocess karo
    cleaned_text = transform_text(email_text)
    
    # 2. Vectorize karo
    transformed_text = vectorizer.transform([cleaned_text]).toarray()
    
    # 3. Predict karo
    prediction = model.predict(transformed_text)[0]

    result = "Spam" if prediction == 1 else "Not Spam (Ham)"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)