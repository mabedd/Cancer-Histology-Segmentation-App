from fastapi import FastAPI, File, UploadFile
import cv2
import io
from starlette.responses import StreamingResponse
import tensorflow as tf
import json
from unet import SegmentationModel

app = FastAPI()

# Load model
model = SegmentationModel().model
model.load_weights("model/model.h5")


@app.post("/")
async def predict(data: UploadFile = File(...)):
    # Convert image to bytes and decode
    image_bytes = await data.read()
    image = tf.io.decode_image(image_bytes)

    # Predict
    prediction = model.predict(tf.expand_dims(image, axis=0))

    return {"Prediction": json.dumps(prediction.tolist())}
