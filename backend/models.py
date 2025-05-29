import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, Wav2Vec2Processor, Wav2Vec2Model, BertForSequenceClassification

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