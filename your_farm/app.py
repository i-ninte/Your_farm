from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import os

app = FastAPI()

# Load the SavedModel
model_dir = "C:/Users/Alif Osman Otoo/Desktop/your_farm/converted_savedmodel/model.savedmodel"
model = tf.saved_model.load(model_dir)

# List available signatures
for signature in model.signatures:
    print(f"Available signature: {signature}")

# Function to perform inference
infer = model.signatures["serving_default"] 

# Load the labels
class_names = open("C:/Users/Alif Osman Otoo/Desktop/your_farm/converted_savedmodel/labels.txt", "r").readlines()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Preprocess the image
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Convert the image to a tensor
    tensor = tf.convert_to_tensor(image)

    # Perform inference
    prediction = infer(tensor)['sequential_3']  # Ensure this is the correct output layer name
    prediction = prediction.numpy()
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return JSONResponse(content={
        "class": class_name,
        "confidence_score": str(np.round(confidence_score * 100))[:-2] + "%"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
