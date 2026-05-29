# 📱 On‑Device AI with Model Optimization

[![Live Demo](https://img.shields.io/badge/Live-Demo-green?logo=huggingface)](https://kallurayaankit-on-device-ai.hf.space)

A complete edge‑AI pipeline that takes a trained model, compresses it with INT8 quantization, benchmarks performance, runs real‑time video inference, performs A/B shadow testing, and simulates over‑the‑air (OTA) updates.

---

## 📌 Features

- **Model export** – PyTorch → ONNX (tiny CNN for demonstration)
- **INT8 quantization** – static quantization with calibration data, reducing model size and latency
- **Performance benchmarking** – compare FP32 vs INT8 latency and size
- **Real‑time edge inference** – webcam capture with ONNX Runtime, showing predicted class and latency
- **A/B shadow testing** – run champion (FP32) and challenger (INT8) side‑by‑side
- **OTA update simulation** – local HTTP server as cloud, client checks for new model versions and downloads them
- **Docker packaging** – ready for deployment on resource‑constrained devices

---

## 📁 Project Structure
on-device-ai/
├── models/
│ ├── tiny_detector.onnx # FP32 exported model
│ └── tiny_detector_int8.onnx # INT8 quantized model
├── export_model.py # PyTorch → ONNX export script
├── quantize_model.py # static quantization script
├── benchmark.py # compare FP32 vs INT8
├── video_inference.py # real‑time webcam demo (INT8)
├── ab_shadow_video.py # A/B shadow comparison
├── ota_update.py # OTA client that checks & downloads new models
├── ota_storage/ # mock cloud server folder (ignored by Git)
├── Dockerfile
├── requirements.txt
└── README.md
