import torch
import torch.nn as nn

class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 150 * 150, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TinyDetector()
model.eval()

dummy_input = torch.randn(1, 3, 300, 300)

torch.onnx.export(
    model,
    dummy_input,
    "models/tiny_detector.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=14
)

print("Exported tiny_detector.onnx")
