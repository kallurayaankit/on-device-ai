import cv2
import numpy as np
import onnxruntime as ort
import time

model_path = 'models/tiny_detector_int8.onnx'
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess: resize to 300x300, normalize, CHW
    img = cv2.resize(frame, (300, 300)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # add batch dim

    start = time.time()
    outputs = session.run([output_name], {input_name: img})
    latency = (time.time() - start) * 1000

    pred = outputs[0][0]                         # shape (10,)
    top_class = np.argmax(pred)
    confidence = pred[top_class]

    # Draw a fixed rectangle and label
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (int(w*0.3), int(h*0.3)), (int(w*0.7), int(h*0.7)), (0,255,0), 2)
    cv2.putText(frame, f"Class {top_class}: {confidence:.2f} ({latency:.1f}ms)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Edge Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()