import yaml
import time
import uvicorn
from collections import deque
from pydantic import BaseModel
from fastapi import FastAPI, Request
from model import ClassificationModel
from fastapi.middleware.cors import CORSMiddleware

class TextData(BaseModel):
    comment: str

## Setting Config and Model
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.Loader)

model = ClassificationModel(config)

## Start fastapi server.
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classification")
async def classifying(request : TextData):
    hate, token_output, tokenized_sentence = model.predict(request.comment)
    
    data = []
    if token_output != None:
        for i in range(len(token_output)):
            data.append({
                "x": tokenized_sentence[i],
                "y": token_output[i],
            })
    
    result = [{
        "data": data,
        "id": "Hate Score",
    }]
    
    result = {
        "is_hate": hate,
        "chart_data": result,
    }
    
    print(result)
    
    return result
