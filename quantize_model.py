import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import numpy as np

class DummyDataReader(CalibrationDataReader):
    def __init__(self, num_samples=100):
        self.iter = iter([{"input": np.random.randn(1, 3, 300, 300).astype(np.float32)} for _ in range(num_samples)])
    def get_next(self):
        return next(self.iter, None)

print("Running static quantization...")
quantize_static(
    model_input="models/tiny_detector.onnx",
    model_output="models/tiny_detector_int8.onnx",
    calibration_data_reader=DummyDataReader(),
    weight_type=QuantType.QInt8,      # <-- QInt8 for weights
    activation_type=QuantType.QInt8   # <-- QInt8 for activations
)
print("✅ Quantized model saved.")