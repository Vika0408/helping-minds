from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, Wav2Vec2Processor, Wav2Vec2Model, BertForSequenceClassification
import google.generativeai as genai
import librosa
import os
import logging
import numpy as np
from prompt import generate_prompt
import openai

# ------------------ Configuration ------------------
logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Initialize Flask ------------------
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:1234"], allow_headers=["Content-Type"])

# ------------------ Load Tokenizer, Processor & Gemini ------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

with torch.no_grad():
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

wav2vec_model.eval()

# Freeze Wav2Vec model parameters
for param in wav2vec_model.parameters():
    param.requires_grad = False

# Load Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY as an environment variable.")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# ------------------ Define Models ------------------
class SentimentModel(nn.Module):
    def __init__(self, num_labels):
        super(SentimentModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask=attention_mask).logits

class EmotionModel(nn.Module):
    def __init__(self, input_size=120, hidden_size=128, num_layers=2, num_classes=8, dropout_prob=0.3):
        super(EmotionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # shape: (batch, seq_len, hidden*2)
        out = lstm_out[:, -1, :]    # Take last time step's output

        # Fully connected layers
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.output(out)
        return out


# ------------------ Load Full Models ------------------
# Initialize and load sentiment model
sentiment_model = SentimentModel(num_labels=3) 
sentiment_model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location=device))
sentiment_model.to(device).eval()

# Initialize and load emotion model
emotion_model = EmotionModel()
emotion_model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
emotion_model.to(device).eval()

# ------------------ Inference Functions ------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = sentiment_model(inputs["input_ids"], inputs["attention_mask"])  # You already return logits
        
    return torch.argmax(logits, dim=1).item()

def predict_emotion(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)  # Wav2Vec expects 16kHz
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        features = wav2vec_model(input_values).last_hidden_state  # [batch, seq, 768]
        logits = emotion_model(features)
        predicted_emotion = torch.argmax(logits, dim=1).item()

    return predicted_emotion


def generate_reply(user_message, sentiment, emotion):
    try:
        prompt = generate_prompt(user_message, sentiment, emotion)
        logging.info(f"Generated prompt: {prompt}")

        response = gemini_model.generate_content(prompt)
        logging.info(f"Gemini raw response: {response}")

        if response and hasattr(response, "text") and response.text.strip():
            return response.text.strip()

        return (
            "I'm here for you. Sometimes, just talking about how you feel can help. "
            "Would you like to share more about what's on your mind?"
        )

    except Exception as e:
        logging.error(f"Gemini generation failed: {e}", exc_info=True)
        return (
            "I’m here to support you, but I’m having trouble generating a response right now. "
            "You're not alone—let's talk about what’s on your mind."
        )


# ------------------ API Routes ------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        text = data.get("message", "").strip()
        
        if not text:
            return jsonify({"error": "Message is empty"}), 400

        sentiment = predict_sentiment(text)
        # If the input is audio, process it for emotion
        emotion = predict_emotion(text) if data.get("is_audio", False) else None

        reply = generate_reply(text, sentiment, emotion)

        return jsonify({
            "reply": reply,
            "sentiment": sentiment,
            "emotion": emotion
        })
    
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK"}), 200

# ------------------ Run Flask Server ------------------
if __name__ == "__main__":
    app.run(debug=True)
