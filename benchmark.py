import onnxruntime as ort
import numpy as np
import time
import os

models = [
    ('FP32', 'models/tiny_detector.onnx'),
    ('INT8', 'models/tiny_detector_int8.onnx')
]

for name, path in models:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    session = ort.InferenceSession(path)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 300, 300).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # Benchmark
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        session.run(None, {input_name: dummy})
    avg_latency = (time.time() - start) / iterations * 1000
    print(f"{name}: {avg_latency:.2f} ms")

# Compare sizes
fp32_size = os.path.getsize(models[0][1]) / 1024
int8_size = os.path.getsize(models[1][1]) / 1024
print(f"\nFP32 size: {fp32_size:.1f} KB")
print(f"INT8 size: {int8_size:.1f} KB")
print(f"Size reduction: {(1 - int8_size/fp32_size)*100:.1f}%")