from fastapi import FastAPI
from model import GenerationModel, Hatespeech
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = GenerationModel()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/purificate")
async def generation(hatespeech: Hatespeech):

    sentence = hatespeech.sentence
    purificated = model.purification(sentence)

    return {"purificated": purificated}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=30001, reload=True)
