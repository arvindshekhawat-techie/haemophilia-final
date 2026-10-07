import torch
import torch.nn as nn


class ElbowLSTM(nn.Module):
    def __init__(
        self,
        input_size=22,   # MUST match your features
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ):
        super(ElbowLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """

        out, _ = self.lstm(x)

        # 🔥 IMPORTANT: last timestep
        out = out[:, -1, :]

        out = self.fc(out)

        return out