import torch.nn.functional as F
import torch.nn as nn
import torch
class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        """CRNN model for OCR.
        Args:
            num_classes: Number of classes (including blank).
            hidden_size: Size of the LSTM hidden state.
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.map_to_seq = nn.Linear(512, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Accepts (B,1,80,W) and outputs CTC‑ready log‑probs (T, B, C).
        We **average over the height dimension** instead of squeezing so we
        never hit a dim‑mismatch even if `AdaptiveAvgPool2d` leaves H>1 on
        some PyTorch builds.
        """
        feats = self.cnn(x)              # (B, 512, H', W')
        feats = feats.mean(2)            # collapse H' → (B, 512, W')
        feats = feats.permute(0, 2, 1)   # (B, W', 512)
        seq, _ = self.rnn(self.map_to_seq(feats))
        logits = self.classifier(seq)    # (B, W', C)
        log_probs = F.log_softmax(logits, dim=-1)  # (B,W',C)
        return log_probs.permute(1, 0, 2)  