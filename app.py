from flask import Flask, request, render_template
import torch
import torch.nn as nn
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Initialize Flask application
app = Flask(__name__)

# Download stopwords if not already done
# nltk.download('stopwords')
stop = set(stopwords.words('english'))

# Define the LSTM model class
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob):
        super(LSTMClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout_prob, 
                            batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        embedded = self.fc(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        output = self.fc_out(lstm_out)
        return output

# Load the model and TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Get the correct input_dim from the vectorizer
input_dim = len(vectorizer.get_feature_names_out())  # Number of features from TfidfVectorizer

model = LSTMClassifier(input_dim=input_dim, embedding_dim=100, hidden_dim=128, output_dim=2, n_layers=2, dropout_prob=0.5)
model.load_state_dict(torch.load('spam_lstm_model.pth', map_location=torch.device('cpu')))
model.eval()

# Preprocess text function
def preprocess_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub('<.*?>', ' ', sentence)
    sentence = re.sub(r'(http|https)://[^\s]*', 'httpaddr', sentence)
    sentence = re.sub(r'[?|!|\'|"|#]', '', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]', ' ', sentence)
    sentence = re.sub(r'[^\s]+@[^\s]+.com', 'emailaddr', sentence)
    sentence = re.sub(r'[0-9]+', 'number', sentence)
    sentence = re.sub(r'[$]+', 'dollar', sentence)
    sentence = re.sub(r'[^a-zA-Z0-9]', ' ', sentence)
    words = [word for word in sentence.split() if word not in stop]
    return ' '.join(words)

# Function to extract most relevant keywords using TF-IDF
def get_spam_keywords(text):
    vectorized_text = vectorizer.transform([text]).toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    
    # Get indices of the top 5 words with the highest TF-IDF scores
    important_indices = vectorized_text.argsort()[-5:][::-1]  # Top 5 highest values
    important_keywords = [feature_names[i] for i in important_indices if vectorized_text[i] > 0]
    return important_keywords

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['email']
    processed_text = preprocess_text(input_text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    input_tensor = torch.tensor(vectorized_text, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        spam_prob = probabilities[1] * 100
        ham_prob = probabilities[0] * 100
        result = 'Spam' if probabilities[1] > probabilities[0] else 'Ham'
        
        # Extract relevant keywords only if the email is detected as spam
        spam_keywords = []
        if result == 'Spam':
            spam_keywords = get_spam_keywords(processed_text)

    return render_template('index.html', 
                           prediction_text=f'This email is: {result} with {spam_prob:.2f}% spam and {ham_prob:.2f}% ham',
                           spam_keywords=spam_keywords)

if __name__ == "__main__":
    app.run(debug=True)
