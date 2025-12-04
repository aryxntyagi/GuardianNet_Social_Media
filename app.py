import pickle
import re
import string
import joblib
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from flask import Flask, request, render_template
import numpy as np

def wordopt(text):
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub("\\\\W", " ", text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

vectorizer = pickle.load(open("models/vectorizer.pkl", 'rb'))

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- BERT Classification Model Wrapper ---
class BertClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: fake (0) and real (1)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        fc1_output = self.relu(self.fc1(pooled_output))
        logits = self.fc2(fc1_output)
        return logits

# --- BERT Model Loading ---
try:
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_base = BertModel.from_pretrained('bert-base-uncased')
    
    # Create the classifier wrapper
    bert_model = BertClassifier(bert_base)
    
    # Load the pre-trained weights (this will load BERT + fc1 and fc2 layers)
    state_dict = torch.load("models/c2_new_model_weights.pt", map_location=device, weights_only=False)
    bert_model.load_state_dict(state_dict, strict=False)  # strict=False allows mismatched keys
    
    bert_model.to(device)
    bert_model.eval()
    
    BERT_AVAILABLE = True
except Exception as e:
    print(f"Warning: BERT model could not be loaded: {e}")
    import traceback
    traceback.print_exc()
    BERT_AVAILABLE = False
    bert_tokenizer = None
    bert_model = None

models = {
    "lr": pickle.load(open("models/model_lr.pkl", 'rb')),
    "dt": pickle.load(open("models/model_dt.pkl", 'rb')),
    "gb": pickle.load(open("models/model_gb.pkl", 'rb'))
}

# --- Hate-speech / cyberbullying models (joblib) ---
# Expecting these files to be in the `models/` folder:
# label_encoder.joblib, logreg_model.joblib, tfidf_vectorizer.joblib
try:
    label_encoder = joblib.load("models/label_encoder.joblib")
    logreg_hate = joblib.load("models/logreg_model.joblib")
    tfidf_hate_lr = joblib.load("models/tfidf_vectorizer.joblib")
except Exception:
    # If any hate-speech artifacts are missing, set to None and handle at runtime
    label_encoder = None
    logreg_hate = None
    tfidf_hate_lr = None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def predict_with_bert(text):
    """Predict using BERT model"""
    if not BERT_AVAILABLE:
        return None, None
    
    try:
        # Tokenize
        inputs = bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            logits = bert_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids']
            )
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max().item()
        
        # Get prediction: 0 = Real, 1 = Fake (or vice versa, may need to swap)
        prediction = torch.argmax(logits, dim=1).item()
        return prediction, logits, confidence
    except Exception as e:
        print(f"BERT prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        news_text = request.form['news_text']
        model_choice = request.form['model_selector']

        # Handle BERT model
        if model_choice == 'bert':
            result = predict_with_bert(news_text)
            if result[0] is None:
                result = "BERT model is not available."
                prediction_class = "error"
            else:
                prediction, _, confidence = result
                # Model was trained with: 0=Fake, 1=Real (based on bias toward class 1)
                if prediction == 0:
                    result = f"This looks like a Fake News article. (Confidence: {confidence:.2%})"
                    prediction_class = "fake-news"
                else:
                    result = f"This looks like a Real News article. (Confidence: {confidence:.2%})"
                    prediction_class = "real-news"
        else:
            # Original TF-IDF based models
            processed_text = wordopt(news_text)
            vectorized_text = vectorizer.transform([processed_text])

            selected_model = models[model_choice]
            prediction = selected_model.predict(vectorized_text)[0]

            if prediction == 1:
                result = "This looks like a Real News article."
                prediction_class = "real-news"
            else:
                result = "This looks like a Fake News article."
                prediction_class = "fake-news"
        
        return render_template(
            'index.html', 
            prediction_text=result,
            prediction_class=prediction_class,
            news_text=news_text,
            model_selected=model_choice
        )


@app.route('/predict_hate', methods=['POST'])
def predict_hate():
    if request.method == 'POST':
        hate_text = request.form.get('hate_text', '')
        model_choice = request.form.get('hate_model_selector', 'logreg')

        # Basic preprocessing using the existing helper
        processed = wordopt(hate_text)

        # Ensure models loaded
        if logreg_hate is None or tfidf_hate_lr is None or label_encoder is None:
            error_msg = "Hate-speech model files are missing on the server."
            return render_template('index.html', hate_prediction_text=error_msg, hate_prediction_class='error', hate_text=hate_text)

        vect = tfidf_hate_lr.transform([processed])
        model = logreg_hate

        pred = model.predict(vect)
        try:
            label = label_encoder.inverse_transform(pred)[0]
        except Exception:
            label = str(pred[0])

        # Friendly message
        if str(label).lower() in ['hate', 'abusive', 'offensive', 'toxic', 'cyberbullying']:
            result = f"This text is predicted as: {label}"
            prediction_class = 'hate'
        else:
            result = f"This text is predicted as: {label}"
            prediction_class = 'no-hate'

        # Render the same index page, provide hate-speech context (leave news fields blank)
        return render_template(
            'index.html',
            hate_prediction_text=result,
            hate_prediction_class=prediction_class,
            hate_text=hate_text,
            hate_model_selected=model_choice,
            news_text='',
            model_selected=None,
            prediction_text=None,
            prediction_class=None
        )

if __name__ == '__main__':
    app.run(debug=True)