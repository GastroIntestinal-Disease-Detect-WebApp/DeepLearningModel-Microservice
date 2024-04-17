from fastapi import FastAPI
import uvicorn
app = FastAPI()
from pydantic import BaseModel
import keras
import os

import keras
import os
import cv2
import numpy as np
import requests
import PIL.Image
from io import BytesIO

labels = {0: 'dyed-lifted-polyps',
 1: 'dyed-resection-margins',
 2: 'esophagitis',
 3: 'normal-cecum',
 4: 'normal-pylorus',
 5: 'normal-z-line',
 6: 'polyps',
 7: 'ulcerative-colitis'}

H = 100
W = 100

def read_image(x):
    x = cv2.resize(x,(H,W))
    x = x/255.0
    x = x.astype(np.float32)
    print(x.shape)
    return x

VGG19 = keras.models.load_model(os.path.join("Weights","model.keras"))


class Image(BaseModel):
  Link:str


@app.post("/getInference")
def getInference(image:Image):
    res  = requests.get(image.Link)

    if res.status_code == 200:
        image_data = BytesIO(res.content)
        img = PIL.Image.open(image_data)
        img = np.array(img)
        np_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = read_image(np_array)
        x = np.expand_dims(x,0)
        res = VGG19.predict(x)
        return {"Diseases":labels[np.argmax(res)]}
 

    else:
        print("Failed to fetch the image")

    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3000)