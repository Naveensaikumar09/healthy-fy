# from fastapi import FastAPI
# from pydantic import BaseModel
# import torch
# from transformers import DistilBertTokenizer
# from model_def import DistilBertRegressor

# app = FastAPI()

# # Load tokenizer and model once at startup
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# model = DistilBertRegressor()
# model.load_state_dict(torch.load("model/healthyfy_model.pth", map_location=torch.device("cpu")))
# model.eval()

# # Input model for request body
# class ParagraphRequest(BaseModel):
#     paragraph: str

# @app.post("/predict_traits")
# async def predict_traits(request: ParagraphRequest):
#     # Tokenize input text
#     inputs = tokenizer(request.paragraph, return_tensors="pt", padding=True, truncation=True)

#     # Disable gradient calculation
#     with torch.no_grad():
#         outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
#         predictions = outputs.squeeze().tolist()

#     traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

#     return {trait: round(score, 3) for trait, score in zip(traits, predictions)}


from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer
from model_def import DistilBertRegressor
from lstm_def import LSTMClassifier
from fastapi.middleware.cors import CORSMiddleware

import pickle
import numpy as np

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:5173"] for safety
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Load First Model (Personality Traits) ----------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
trait_model = DistilBertRegressor()
trait_model.load_state_dict(torch.load("model/healthyfy_model.pth", map_location=torch.device("cpu")))
trait_model.eval()

# --------- Load Second Model (LSTM Risk Classifier) ----------
lstm_model = LSTMClassifier(input_size=5, hidden_size=48, num_layers=1, output_size=3)
checkpoint = torch.load("model/model_2.pth", map_location=torch.device("cpu"), weights_only=False)

lstm_model.load_state_dict(checkpoint['model_state_dict'])
lstm_model.eval()

# Load scaler and label encoder
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

label_classes = checkpoint['label_encoder']

# --------- API Input Schema ----------
class ParagraphRequest(BaseModel):
    paragraph: str

# --------- API Endpoint ----------
@app.post("/predict")
async def predict(request: ParagraphRequest):
    # --- First Model: Predict Traits ---
    inputs = tokenizer(request.paragraph, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        trait_outputs = trait_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        traits = trait_outputs.squeeze().tolist()

    # Trait Names
    trait_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    trait_dict = {name: round(score, 3) for name, score in zip(trait_names, traits)}

    # --- Second Model: Predict Risk Label ---
    trait_array = np.array(traits).reshape(1, -1)
    scaled_traits = scaler.transform(trait_array)
    input_tensor = torch.tensor(scaled_traits, dtype=torch.float32).unsqueeze(1)  # shape: (1, 1, 5)

    with torch.no_grad():
        lstm_output = lstm_model(input_tensor)
        predicted_class = torch.argmax(lstm_output, dim=1).item()
        risk_label = label_classes[predicted_class]

    return {
        **trait_dict,
        "Risk": risk_label
    }
