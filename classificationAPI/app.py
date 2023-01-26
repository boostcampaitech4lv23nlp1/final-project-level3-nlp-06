import yaml
import uvicorn
from fastapi import FastAPI, Request
from model import ClassificationModel

## Setting Config and Model
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.Loader)
model = ClassificationModel(config)

## Start fastapi server.
app = FastAPI()

@app.get("/classification")
async def classifying(request: Request, comments: str):
    hate, token_output, tokenized_sentence = model.predict(comments)
    result = {
        "is_hate": hate,
        "token_hate": token_output,
        "tokenized_sentence": tokenized_sentence
    }
    return result

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=30001)
    