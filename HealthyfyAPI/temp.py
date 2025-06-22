import torch

model_path = "model/healthyfy_model.pth"
data = torch.load(model_path, map_location="cpu")
print(type(data))
