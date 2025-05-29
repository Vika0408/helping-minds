# Helping Minds 🧠💬

**Helping Minds** is a mental health assistant web application that uses AI to help users understand and manage their emotions. The system performs real-time sentiment analysis on text and emotion detection from voice, enabling a smart chatbot that responds empathetically. It also provides a diary and to-do list for mental wellness tracking.

---

## 🌟 Features

- 🎤 **Emotion Detection from Voice**  
  Uses an LSTM model trained on RAVDESS dataset with MFCC audio features.

- 💬 **Sentiment Analysis from Text**  
  Utilizes a BERT-based model fine-tuned on the MELD dataset.

- 🧠 **AI-Powered Chatbot**  
  Responds empathetically based on detected emotional and sentiment states.

- 📓 **Diary**  
  Enables users to log and reflect on their daily mental state.

- ✅ **To-Do List**  
  Helps manage daily tasks to support productivity and mental clarity.

- 🔁 **Frontend-Backend Integration**  
  Real-time communication between React.js frontend and Flask backend.

---

## 🛠️ Tech Stack

- **Frontend**: React.js, Tailwind CSS  
- **Backend**: Flask (Python)  
- **ML Models**:  
  - BERT for sentiment analysis (text)  
  - LSTM with MFCC for emotion detection (audio)  
- **Libraries/Tools**:  
  PyTorch, Hugging Face Transformers, SpeechRecognition, OpenAI/Gemini API, Wav2Vec2

---

## 📁 Project Structure

<pre> mental_health/ 
  ├── backend/
  │ ├── app.py # Main Flask app
  │ ├── emotion_model.py # LSTM emotion detection logic 
  │ ├── sentiment_model.py # BERT sentiment analysis logic 
  │ └── utils/ # Feature extraction and preprocessing 
  ├── frontend/ 
  │ ├── public/ 
  │ └── src/ 
  │ ├── App.js
  │ ├── components/ # UI Components (ChatBot, Diary, etc.)
  │ └── services/ # API handling
  ├── saved_models/ # Trained .pt model files 
  ├── requirements.txt # Python dependencies
  └── README.md </pre>



---

## 🚀 Getting Started Locally

### Prerequisites

- Python 3.9+  
- Node.js 18+  
- npm  
- Git  

---

### Step 1: Clone the Repository


git clone https://github.com/Vika0408/helping-minds.git
cd helping-minds


cd backend
pip install -r requirements.txt
python app.py

cd frontend
npm install
npm start


🧠 AI Models
🎤 Emotion Detection
Model: LSTM

Dataset: RAVDESS

Preprocessing: Extracted MFCC features from audio clips.

Accuracy: Trained and evaluated with cross-validation.

💬 Sentiment Analysis
Model: BERT base (uncased)

Dataset: MELD (Multi-modal EmotionLines Dataset)

Labels: Positive, Neutral, Negative

Fine-Tuning: Using Hugging Face Transformers


📷 Screenshots

![home_screen](https://github.com/user-attachments/assets/3897bc37-e904-4dda-a43c-ac10b522412f)


![services](https://github.com/user-attachments/assets/d1eac262-911f-498a-b3f2-63f4ca573d3c)





