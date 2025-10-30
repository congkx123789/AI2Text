import torch, torch.nn as nn

class CRNNCTC(nn.Module):
    def __init__(self, n_mels=80, vocab_size=40, cnn_channels=128, rnn_hidden=256, rnn_layers=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=cnn_channels, hidden_size=rnn_hidden,
            num_layers=rnn_layers, batch_first=True, bidirectional=True
        )
        self.head = nn.Linear(rnn_hidden*2, vocab_size)

    def forward(self, x, x_lens):
        # x: (B, T, F)
        x = x.transpose(1,2)    # (B, F, T)
        x = self.cnn(x)         # (B, C, T)
        x = x.transpose(1,2)    # (B, T, C)
        x, _ = self.rnn(x)      # (B, T, 2H)
        logits = self.head(x)   # (B, T, V)
        return logits.transpose(0,1), x_lens  # to (T, B, V)
