import random
import logging
import datetime
from asyncio import run
from mongo import MongoDB
from fastapi import FastAPI, Request
from pymongo import MongoClient
from datetime import date, timedelta
from request_form import LabeledData, CheckedData, RewardData, PostData, CommentData
from fastapi.middleware.cors import CORSMiddleware


## Static Parameters
DATA_LENGTH = 72706
reward_check = [0] * 601


## Connections
mongo = MongoDB()
app = FastAPI()
logger = logging.getLogger()

## Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## APIs
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/api/count/visit")
async def visit_count():
    
    today = date.today()
    out = await mongo.find_one("hate_data", "view_count", {"date": str(today)})
    if out == None:
        out = await mongo.insert_one("hate_data", "view_count", {"date": str(today), "count": 1})
    else:
        out = await mongo.update_one("hate_data", "view_count", {"date": str(today)}, {"count": out["count"] + 1})
    
    return {"result": "ok"}


@app.get("/api/get_text")
async def get_text():
    ## Get random data which status is not 0
    while True:
        id = random.randrange(0, DATA_LENGTH)
        info = {"_id": id}
        
        data = await mongo.find_one("hate_data", "hate_data", info)
        if data["status"] == 0:
            break
        
    text = data["text"]
    return {"text": text, "text_id": id}


@app.put("/api/labeling")
async def labeling(request : LabeledData):
    ## Get request
    if request.check1:
        check_box = 1
    elif request.check2:
        check_box = 2
    else:
        check_box = 3
        
    id = request.id
    text = request.text
    target = request.labeledText
    nick_name = request.nickName
    
    ## Update the hate data
    info = {"_id": id}
    data = await mongo.update_one("hate_data", "hate_data", info, {"status": 1})
    
    ## Store to need check collection
    info = {
        "_id": id,
        "text": text,
        "target": target,
        "label": check_box,
        "nick_name": nick_name
    }
    data = await mongo.insert_one("hate_data", "need_check_data", info)
    
    ## Labeled count
    today = date.today()
    out = await mongo.find_one("hate_data", "labeled_count", {"date": str(today)})
    if out == None:
        out = await mongo.insert_one("hate_data", "labeled_count", {"date": str(today), "count": 1})
    else:
        out = await mongo.update_one("hate_data", "labeled_count", {"date": str(today)}, {"count": out["count"] + 1})
    
    ## Count need check
    out = await mongo.find_one("hate_data", "check_count", {"id": -1})
    out = await mongo.update_one("hate_data", "check_count", {"id": -1}, {"count": out["count"] + 1})
    
    ## Get random data which status is not 0
    while True:
        id = random.randrange(0, DATA_LENGTH)
        info = {"_id": id}
        
        data = await mongo.find_one("hate_data", "hate_data", info)
        if not data["status"]:
            break
        
    text = data["text"]
    return {"text": text, "text_id": id}


@app.get("/api/get_need_check")
async def get_need_check():
    out = await mongo.find_many("hate_data", "need_check_data", data={}, limit=30)
    
    response = {}
    for i in range(len(out)):
        response["data"+str(i)] = out[i]
    
    return response


@app.post("/api/permit")
async def permit(request: CheckedData):
    id = request.id
    text = request.text
    target = request.target
    nick_name = request.nick_name
    label = request.label
    
    ## 1. delete data in need_check
    out = await mongo.delete_one("hate_data", "need_check_data", {"_id": id})
    
    ## 2. update in hate data -> 2 + update target
    if label == 1:
        status = 3
    elif label == 2:
        status = 4
    else:
        status = 2
    print(status)
    out = await mongo.update_one("hate_data", "hate_data", {"_id": id}, {"target": target, "status": status})
    
    ## 3. user info update
    out = await mongo.find_one("hate_data", "user_info", {"nick_name": nick_name})
    if out == None:
        out = await mongo.insert_one("hate_data", "user_info", {"nick_name": nick_name, "count": 1})
    else:
        out = await mongo.update_one("hate_data", "user_info", {"nick_name": nick_name}, {"count": out["count"] + 1})
    
    ## 4. labeled count by date
    today = date.today()
    out = await mongo.find_one("hate_data", "target_count", {"date": str(today)})
    if out == None:
        out = await mongo.insert_one("hate_data", "target_count", {"date": str(today), "count": 1})
    else:
        out = await mongo.update_one("hate_data", "target_count", {"date": str(today)}, {"count": out["count"] + 1})
    print("ok")
    
    ## 5. Delete the count
    out = await mongo.find_one("hate_data", "check_count", {"id": -1})
    out = await mongo.update_one("hate_data", "check_count", {"id": -1}, {"count": out["count"] - 1})
    
    return {"result": "ok"}


@app.post("/api/reject")
async def reject(request: CheckedData):
    id = request.id
    text = request.text
    target = request.target
    nick_name = request.nick_name
    
    ## 1. delete data in need_check
    out = await mongo.delete_one("hate_data", "need_check_data", {"_id": id})
    ## 2. update in hate data -> 0 original
    out = await mongo.update_one("hate_data", "hate_data", {"_id": id}, {"target": target, "status": 0})
    ## 3. Delete the count
    out = await mongo.find_one("hate_data", "check_count", {"id": -1})
    out = await mongo.update_one("hate_data", "check_count", {"id": -1}, {"count": out["count"] - 1})
    
    print("reject")
    
    return {"result": "reject"}


@app.get("/api/reward_data")
async def reward_data():
    count = 0
    while True:
        id = random.randrange(0, 600)
        info = {"_id": id}
        
        data = await mongo.find_one("reward_model_data", "data", info)
        count += 1

        if data["status"] == 0:
            break
        if count == 2000:
            break
    
    if count == 2000:
        return {
            "_id": -1,
            "text": "끝났거나, 다시 새로고침이 필요하거나",
            "data1": "정말 끝이라는 소리임",
            "data2": "아니면 새로고침이 필요할지도?",
            "data3": "두번 새로고침했는데도 안되면 끝난거임",
            "data4": "",
        }
        
    return data
    
@app.post("/api/post_reward_data")
async def post_reward_data(request: RewardData):
    id = request.id
    reward1 = request.reward1
    reward2 = request.reward2
    reward3 = request.reward3
    reward4 = request.reward4
    data = {
        "_id": id,
        "reward1": reward1,
        "reward2": reward2,
        "reward3": reward3,
        "reward4": reward4,
    }

    out = await mongo.update_one("reward_model_data", "data", {"_id": id}, {"status": 1})
    
    out = await mongo.insert_one("reward_model_data", "checked_data", data)
    print("reward data: insert ok")
    return {"result": "ok"}


@app.get("/api/count/get_top_user")
async def get_top_user():
    ## 1. get user infomation
    out = await mongo.find_many("hate_data", "user_info", {}, "count", 10)
    out = list(out)
    out = out[-10:]
    
    response = []
    for item in out:
        response.append({"user": item["nick_name"], "data_count": item["count"]})
        
    return {"data": response}

@app.get("/api/dashboard_data")
async def get_dashboard_data():
    response = {}
    
    today = date.today()
    
    ## 1. get today data
    out = await mongo.find_one("hate_data", "labeled_count", {"date": str(today)})
    if out == None:
        out = await mongo.insert_one("hate_data", "labeled_count", {"date": str(today), "count": 0})
        today_data = 0
    else:
        today_data = out["count"]
     
    response["today_data"] = today_data
    
    ## 2. get total data
    daily_data = []
    today = date.today() - timedelta(7)
    for i in range(8):
        now = today + timedelta(i)
        out = await mongo.find_one("hate_data", "labeled_count", {"date": str(now)})
        daily_data.append({"x": str(now), "y": out["count"]})
    
    response["daily_data"] = [{"id": "Daily Collected", "data": daily_data}]
    
    ## 3. get top user
    out = await mongo.find_many("hate_data", "user_info", {}, "count", 10)
    out = list(out)
    out = out[-10:]
    
    top_user = []
    for item in out:
        top_user.append({"user": item["nick_name"], "data_count": item["count"]})
    
    response["top_user"] = top_user
    
    ## 4. get total need check data
    out = await mongo.find_one("hate_data", "check_count", {"id": -1})
    check_count = out["count"]
    
    response["check_count"] = check_count
    
    ## 5. get today hit
    today = date.today()
    out = await mongo.find_one("hate_data", "view_count", {"date": str(today)})
    today_hit = out["count"]
    
    response["today_hit"] = today_hit
    
    return response

@app.get("/api/post")
async def get_posts():
    db = mongo.client["poster"]
    collection = db["post_info"]
    
    output = await mongo.find_one("poster", "post_info", {"name": "counter"})
    output = await collection.find({}, {"comments": 0}).to_list(output["count"])
    
    return {"datas": output}

@app.post("/api/one_post")
async def get_one_post(request : PostData):
    id = request.id
    output = await mongo.find_one("poster", "post_info", {"_id": id})
    
    return output

@app.post("/api/post_comment")
async def post_comment(request : CommentData):
    id = request.id
    comment = request.comment
    fixed = request.fixed
    today = str(date.today())
    
    data = {
        "comment": comment,
        "date": today,
        "fixed": fixed,
    }
    
    output = await mongo.push_data("poster", "post_info", {"_id": id}, {"comments": data})
    
    return {"test": "test"}
