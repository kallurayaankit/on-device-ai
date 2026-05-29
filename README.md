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
