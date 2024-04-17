from fastapi import FastAPI
import os
from fastapi.staticfiles import StaticFiles
import random
import requests

app = FastAPI()
app.mount("/images", StaticFiles(directory="static/Images"), name="images")

@app.get("/")
def Home():
    return  "<h1>hi</h1>"

@app.get("/getImages")
def getImages():

    list_dir = []
    for i in os.listdir(os.path.join("static","Images")):
          list_dir.append(i)
    
    num = random.randint(0,len(list_dir)-1)
    return list_dir[num]

@app.get("/sendImageLink")
def sendImageLink():
    
    list_dir = []
    for i in os.listdir(os.path.join("static","Images")):
          list_dir.append(i)
    
    num = random.randint(0,len(list_dir)-1)
    image = list_dir[num]
    res = requests.post("http://localhost:3000/getInference",json={"Link":f"http://127.0.0.1:8000/images/{image}"})
    return res.json()


     
     


