import yaml
import time
import uvicorn
from collections import deque
from fastapi import FastAPI, Request
from model import ClassificationModel
from fastapi.middleware.cors import CORSMiddleware

## Setting Config and Model
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.Loader)

models = deque([(ClassificationModel(config), i) for i in range(4)])
# model = ClassificationModel(config)

## Start fastapi server.
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/classification")
async def classifying(request: Request, comments: str):
    while len(models) == 0:
        time.sleep(0.0001)
    
    model, index = models.pop()

    hate, token_output, tokenized_sentence = model.predict(comments)
    result = {
        "is_hate": hate,
        "token_hate": token_output,
        "tokenized_sentence": tokenized_sentence
    }
    models.appendleft((model, index))
    
    return result

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
    