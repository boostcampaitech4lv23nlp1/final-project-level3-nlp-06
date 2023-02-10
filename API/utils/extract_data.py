import logging
import pandas as pd
import asyncio, time
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient


client = MongoClient(host="localhost", port=27017)

data = None

db = client["hate_data"]
collection = db["hate_data"]

data = collection.find({})
ids = []
texts = []
targets = []
for item in data:
    id = item["_id"]
    text = item["text"]
    target = item["target"]
    status = item["status"]
    if status == 2 and len(text) < 200:
        ids.append(id)
        texts.append(text)
        targets.append(target)
        
        
df = pd.DataFrame({"id": ids, "text": texts, "target": targets})
df.to_csv("paired_data.csv", index=False)