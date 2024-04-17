from fastapi import FastAPI
import uvicorn
app = FastAPI()
from pydantic import BaseModel
import keras
import os
import httpx 
import os
import cv2
import numpy as np
import requests
import PIL.Image
from io import BytesIO
from schemas import ImageLinkInput

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


@app.post("/perform_prediction")
async def getInference(image:ImageLinkInput):
    res  = requests.get(image.image_link)

    if res.status_code == 200:
        image_data = BytesIO(res.content)
        img = PIL.Image.open(image_data)
        img = np.array(img)
        np_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = read_image(np_array)
        x = np.expand_dims(x,0)
        res = VGG19.predict(x)
        
        response = await update_response_in_db(labels[np.argmax(res)],image.image_link)
        print(response)

    else:
        print("Failed to fetch the image")

async def update_response_in_db(response_from_model, image_link):
    api_url = "http://127.0.0.1:8000/update-model-response"
    data = {
        "image_link":image_link, 
        "response_from_model":response_from_model
        }
    
    async with httpx.AsyncClient() as client:
        try:
            await client.put(api_url, json=data)
        except Exception as e:
            print(f"Failed to send request: {e}")

if __name__ == "__main__":
    uvicorn.run("ml_server:app",reload=True,port=8003)