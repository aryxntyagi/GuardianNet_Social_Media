import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

device = torch.device('cpu')

class BertClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        fc1_output = self.relu(self.fc1(pooled_output))
        logits = self.fc2(fc1_output)
        return logits

# Load model
print("Loading BERT model...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_base = BertModel.from_pretrained('bert-base-uncased')
bert_model = BertClassifier(bert_base)

state_dict = torch.load("models/c1_fakenews_weights_BERT.pt", map_location=device)
bert_model.load_state_dict(state_dict, strict=False)
bert_model.eval()

# Test with different texts
test_texts = [
    "This is definitely fake news",
    "Breaking: Real news from trusted sources",
    "Unknown story about politics",
]

print("\n" + "="*60)
print("Testing with ORIGINAL labels (0=Real, 1=Fake):")
print("="*60)
for text in test_texts:
    inputs = bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        logits = bert_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
    
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    
    print(f"\nText: {text}")
    print(f"Logits: {logits.squeeze().detach()}")
    print(f"Probabilities: Real={probs[0,0]:.4f}, Fake={probs[0,1]:.4f}")
    print(f"Prediction (0=Real, 1=Fake): {prediction} -> {'Fake' if prediction == 1 else 'Real'}")

print("\n" + "="*60)
print("Testing with SWAPPED labels (0=Fake, 1=Real):")
print("="*60)
for text in test_texts:
    inputs = bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        logits = bert_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
    
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    # SWAP THE LOGIC
    swapped_prediction = 1 - prediction
    
    print(f"\nText: {text}")
    print(f"Logits: {logits.squeeze().detach()}")
    print(f"Probabilities: Fake={probs[0,0]:.4f}, Real={probs[0,1]:.4f}")
    print(f"Swapped Prediction (0=Fake, 1=Real): {swapped_prediction} -> {'Real' if swapped_prediction == 1 else 'Fake'}")
