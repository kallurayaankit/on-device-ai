from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

app = FastAPI()

# Load the quantized model
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read and preprocess the image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((300, 300))
    img_arr = np.array(img).astype(np.float32) / 255.0
    img_arr = np.transpose(img_arr, (2, 0, 1))   # HWC → CHW
    img_arr = np.expand_dims(img_arr, axis=0)     # add batch dim

    # Run inference
    outputs = session.run([output_name], {input_name: img_arr})
    predictions = outputs[0][0]                   # shape (10,)
    top_class = int(np.argmax(predictions))
    confidence = float(predictions[top_class])

    return {
        "predicted_class": top_class,
        "confidence": round(confidence, 4)
    }