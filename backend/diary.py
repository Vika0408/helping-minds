from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import sqlite3
from transformers import BertTokenizer
from models import SentimentModel
import os

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentiment_model = SentimentModel(num_labels=3)
sentiment_model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location=device))
sentiment_model.to(device).eval()

# Create diary DB if not exist
def init_db():
    conn = sqlite3.connect("diary.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = sentiment_model(inputs["input_ids"], inputs["attention_mask"])
    return torch.argmax(logits, dim=1).item()

@app.route("/diary", methods=["POST"])
def save_entry():
    data = request.get_json()
    text = data.get("entry", "").strip()
    if not text:
        return jsonify({"error": "Empty diary entry"}), 400
    sentiment = predict_sentiment(text)
    conn = sqlite3.connect("diary.db")
    conn.execute("INSERT INTO entries (text, sentiment) VALUES (?, ?)", (text, sentiment))
    conn.commit()
    conn.close()
    return jsonify({"message": "Entry saved", "sentiment": sentiment})

@app.route("/diary", methods=["GET"])
def get_entries():
    conn = sqlite3.connect("diary.db")
    cursor = conn.execute("SELECT id, text, sentiment FROM entries ORDER BY id DESC")
    entries = [{"id": row[0], "text": row[1], "sentiment": row[2]} for row in cursor.fetchall()]
    conn.close()
    return jsonify(entries)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
