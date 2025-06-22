
import torch
import torch.nn as nn
from transformers import DistilBertModel

class DistilBertRegressor(nn.Module):
    def __init__(self):
        super(DistilBertRegressor, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.regressor = nn.Linear(768, 5)
        self.sigmoid = nn.Sigmoid()  # Added

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        logits = self.regressor(hidden_state)
        return self.sigmoid(logits)  # Apply sigmoid to keep values in [0, 1]





# # to run : uvicorn main:app --reload