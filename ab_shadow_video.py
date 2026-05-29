import cv2
import numpy as np
import onnxruntime as ort
import time

# Load the two "brains" – original (champion) and compressed (challenger)
champion = ort.InferenceSession('models/tiny_detector.onnx')
challenger = ort.InferenceSession('models/tiny_detector_int8.onnx')

# Both brains expect the same type of image input and give the same type of output
input_name = champion.get_inputs()[0].name
output_name = champion.get_outputs()[0].name

# Start the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image exactly the same way for both brains
    img = cv2.resize(frame, (300, 300)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # rearrange colour channels
    img = np.expand_dims(img, axis=0)            # add batch dimension

    # Let the champion brain think and measure how long it took
    start = time.time()
    champion.run([output_name], {input_name: img})
    lat_champ = (time.time() - start) * 1000   # milliseconds

    # Let the challenger brain think and measure its time
    start = time.time()
    challenger.run([output_name], {input_name: img})
    lat_chal = (time.time() - start) * 1000

    # Show the two times on the screen so you can compare them
    cv2.putText(frame, f"Champion: {lat_champ:.1f} ms   Challenger: {lat_chal:.1f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('A/B Shadow', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()