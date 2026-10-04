import torch
import numpy as np
from src.models.elbow_lstm import ElbowLSTM

class Predictor:
    def __init__(self):
        self.model = ElbowLSTM()
        self.model.load_state_dict(torch.load("models/elbow_lstm.pth"))
        self.model.eval()

    def predict(self, x):
        x = torch.tensor(x).unsqueeze(0).float()
        prob = torch.sigmoid(self.model(x)).item()
        return prob