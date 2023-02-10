import logging
import asyncio, time
from pymongo import MongoClient
#from motor.motor_asyncio import AsyncIOMotorClient


class MongoDB():
    def __init__(self):
        self.client = MongoClient(host="localhost", port=27017)
        self.logger = logging.getLogger()
        
    def insert_one(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = collection.insert_one(data)
        except:
            print("Insert one function have errors")
    
    
    def insert_many(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]

        try:
            output = collection.insert_many(data)
        except:
            print("Insert many function have errors")
           
            
    def delete_one(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]

        try:
            output = collection.delete_one(data)
        except:
            print("Delete one function have errors")
    
    
    def delete_many(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]

        try:
            output = collection.delete_many(data)
        except:
            print("Delete many function have errors")
    
            
    def update_one(self, db_name, collection_name, old_data, new_data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = collection.update_one(old_data, {"$set": new_data})
        except:
            print("Update one function have errors")
            return None
    
    def update_many(self, db_name, collection_name, old_data, new_data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = collection.update_many(old_data, {"$set": new_data})
            return output
        except:
            print("Update many function have errors")
            return None
    
    def find_one(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = collection.find_one(data)
            return output
        except:
            print("Find one function have errors")
            return None
    
    
    def find_many(self, db_name, collection_name, data, sort_key=None, limit=None):
        db = self.client[db_name]
        collection = db[collection_name]
        a = collection.find({}).sort("count")
        print(list(a))

        if sort_key is not None:
            if limit == None:
                output = collection.find(data).sort(sort_key)
            else:
                output = collection.find(data).sort(sort_key).limit(limit)
        else:
            if limit == None:
                output = collection.find(data)
            else:
                output = collection.find(data).limit(limit)
                
        return output

    def find(self,db_name, collection_name, sort_key=None, limit=None):
        db = self.client[db_name]
        collection = db[collection_name]
        

        if sort_key is not None:
            if limit == None:
                output = collection.find().sort(sort_key)
            else:
                output = collection.find().sort(sort_key).limit(limit)
        else:
            if limit == None:
                output = collection.find()
            else:
                output = collection.find().limit(limit)
                
        return output
    
    def count_documents(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = collection.count_documents(data)
            return output
        except:
            print("Count documents function have errors")
            return None
       
        
    def drop_collection(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = collection.drop()
        except:
            print("Drop collection function have errors")
    
    
    def drop_database(self, db_name):
        try:
            self.client.drop_database(db_name)
        except:
            print("Drop database function have errors")
        
     
import pandas as pd
def insert_hate_data(mongo):
    dataset = pd.read_csv("all_hate.csv")
    
    text_list = list(dataset["text"])
    for i in range(len(dataset)):
        now_data = dataset.iloc[i]
        text = text_list[i]
        
        data = {
            "_id": i,
            "text": text,
            "target": "",
            "status": 0,
        }
        
        mongo.insert_one("hate_data", "hate_data", data)
    print("insert ok")

def count_info(mongo):
    mongo.drop_collection("hate_data", "hate_data")
    mongo.drop_collection("hate_data", "test_data")
    mongo.drop_collection("hate_data", "view_count")
    mongo.drop_collection("hate_data", "labeled_count")
    mongo.drop_collection("hate_data", "target_count")
    mongo.drop_collection("hate_data", "need_check_test")
    mongo.drop_collection("hate_data", "user_info")
    db = mongo.client["hate_data"]
    collection = db["view_count"]
    collection = db["labeled_count"]
    collection = db["target_count"]
    collection = db["user_info"]
    
    print("ok")

def initial(mongo):
    count_info(mongo)
    insert_hate_data(mongo)
    print("all ok")
    return

def init_reward_database(mongo):
    mongo.drop_collection("reward_model_data", "data")
    mongo.drop_collection("reward_model_data", "checked_data")
    db = mongo.client["reward_model_data"]
    collection = db["data"]

def insert_reward_data(mongo):
    data = pd.read_csv("reward_sample.csv")
    
    text = list(data["text"])
    data1 = list(data["label1"])
    data2 = list(data["label2"])
    data3 = list(data["label3"])
    data4 = list(data["target"])
    id = list(data["id"])
    
    for i in range(len(data1)):
        now_data = {
            "_id": i,
            "p_id": id[i],
            "text": text[i],
            "data1": data1[i],
            "data2": data2[i],
            "data3": data3[i],
            "data4": data4[i],
            "status": 0,
        }
        mongo.insert_one("reward_model_data", "data", now_data)
    print("reward insert ok")


def fix_reward(mongo):
    db = mongo.client["reward_model_data"]
    collection = db["checked_data"]
    fix = db["data"]
    
    temp = collection.find()
    for item in temp:
        id = item["_id"]
        mongo.update_one("reward_model_data", "data", {"_id": id}, {"status": 1})
    print("fix ok")
    
def reward_data_extraction(mongo):
    db = mongo.client["reward_model_data"]
    collection = db["checked_data"]
    
    temp = collection.find()
    
    ids = []
    reward1s = []
    reward2s = []
    reward3s = []
    reward4s = []
    
    for item in temp:
        id = item["_id"]
        reward1 = item["reward1"]
        reward2 = item["reward2"]
        reward3 = item["reward3"]
        reward4 = item["reward4"]

        ids.append(id)
        reward1s.append(reward1)
        reward2s.append(reward2)
        reward3s.append(reward3)
        reward4s.append(reward4)
        
    data = pd.DataFrame({"_id": ids, "reward1": reward1s, "reward2": reward2s, "reward3": reward3s, "reward4": reward4s})
    data.to_csv("rewarded_data.csv", index=False)


def post_info(mongo):
    name = "poster"
    for i in range(6):
        now = name + str(i)
        mongo.drop_collection("poster", now)
    mongo.drop_collection("poster", "post_info")
    mongo.drop_collection("poster", "comment_info")
    
    ids = [0, 1, 2, 3, 4, 5, 6]
    
    post_title = [
        "배우 이승기, 카이스트 3억 원 또 기부 [연예뉴스 HOT]",
        "tvN ‘장사천재 백사장’, 세계 요식업 시장에 도전장 [연예뉴스 HOT]",
        "중국, 일본인 비자 발급 재개…한국은 해제 안해",
        "尹 대통령 해치겠다 경찰 협박 전화한 50대 男 입건",
        "멜라민 파동 겪고도 또…中초등생 빵먹고 사망, 독극물 중독",
        "이태원 참사 100일 맞아…여야 함께 '국회 추모제' 연다",
        ]
    
    post_text = [
        "가수 겸 배우 이승기가 카이스트(한국과학기술원)에 3억 원 기부를 약속했다. 1일 과학계에 따르면 이승기는 3일 오후 카이스트 분원 캠퍼스에서 ‘카이스트 발전기금 약정식’을 체결한다. 기부 목적은 국가 미래 경쟁력인 과학기술 발전을 이끌어달라는 취지다. 특히 카이스트가 글로벌 대학으로 발돋움할 수 있도록 기부금을 약정할 것으로 알려졌다. 이승기는 전 소속사와 음원료 미정산 등으로 법적 분쟁을 벌이는 상황에서도 최근 두 달 사이 약 30억 원을 기부하며 선행을 이어가는 중이다. 지난해 12월 서울대어린이병원에 20억 원을 쾌척하고 지난달에는 대한적십자사에 5억5000만 원을 기부했다.",
        "요리연구가 백종원과 소녀시대 유리, 배우 이장우, 가수 존박 등이 tvN 예능프로그램 ‘장사천재 백사장’(가제) 출연한다. 2일 tvN은 “백종원이 세계 요식업 시장에 도전하는 새 프로그램을 준비하고 있다”면서 “안전하고 원활한 촬영을 위해 세부 콘셉트 등은 추후 공개 예정”이라고 밝혔다. 이들의 관련 소식은 앞서 이탈리아 현지에서 먼저 전해졌다. 최근 이탈리아 언론 ‘팬페이지’ 등은 “나폴리에서 80년 이상 운영되던 전통 피자 식당이 있던 곳에 한식당이 개업했다”면서 “이 식당이 한국의 예능 촬영과 관련이 있다”고 보도했다.",
        "중국발 입국자에 대한 방역 강화 조처에 맞서 한·일 양국에 대한 비자 발급을 중단했던 중국이 일본에 대해선 보복을 해제했다. 일본은 한국과 달리 애초 방역 조처를 강화하면서도 비자 발급을 중단하진 않아 중국도 ‘과도한 보복’이라는 비난을 받아온 조처를 서둘러 풀어버린 것으로 해석된다.\n\n주일본 중국대사관은 29일 오후 누리집을 통해 “오늘부터 중국 주일 대사관과 총영사관은 일본 국민에 대한 중국의 일반 비자 발급을 재개했다”고 밝혔다. 일반 비자는 외교·공무 등을 제외한 비자를 의미한다.\n\n<아사히신문>은 이 조처의 의미에 대해 “비자 발급이 정지되며 일본 기업의 중국 출장자들의 발이 묶이는 등 중국 산업의 공급망에 타격이 있었다”고 지적했다. 중·일 간 경제 교류가 막히는 영향 등을 우려해 중국이 일본에 대한 비자 발급을 조기에 재개했다고 분석한 것이다. 대사관은 이 조처를 취하는 이유를 따로 밝히진 않았다.",
        "윤석열 대통령을 해치겠다며 경찰을 협박한 50대 남성이 붙잡혀 조사를 받고 있다.\n\n경찰에 따르면 서울 광진경찰서는 3일 자정쯤 택시 안에서 112로 전화를 걸어 “윤석열 대통령을 해치겠다. 지금 용산으로 가고 있다”며 협박한 50대 남성을 협박 혐의로 입건, 조사중이다.\n\n경찰은 이날 오전 서울 광진구 구의동에서 전화를 건 남성으로 김모(55)씨를 특정, 그의 자택 근처에서 임의동행해 조사했다.\n\n김씨는 경찰에서 “술에 취한 상태로 한 말로, 실제 해치려는 의도는 없었다”는 취지로 진술한 것으로 전해졌다.\n\n경찰 관계자는 “경찰의 심야 시간 공조로 상황 대비에 들어간 만큼 위계공무집행방해 혐의를 적용할 수 있을지 여부 등을 추가로 검토하겠다”고 설명했다.",
        "중국에서 빵을 사먹은 초등학생이 독극물 중독으로 사망해 생산업체 관계자들이 대거 체포됐다.\n\n3일(현지시간) 홍성신문 등 중국 현지 매체에 따르면 지난해 9월 광둥성 잔장시 쉬원현에서 발생한 초등생 사망사건 관련 빵 생산업체 대표 등 8명이 체포됐다. 학생들 사망 원인이 독극물 성분 중독으로 드러난 데 따른 조치다.\n\n당시 10세였던 피해 초등학생은 학교 앞 매점에서 9위안(약 1500원)짜리 빵을 사먹고 중독 증세를 보여 병원 이송됐으나 20여일만에 사망했다.\n\n학생 부모는 “딸이 아침을 안먹어 학교 앞 매점에서 빵과 우유를 사서 등교했다. 평소 건강했다”고 증언했다.\n\n2008년 멜라민 분유 파동을 겪은 중국 사회에서는 이번 사건으로 철저한 진상 규명을 요구하는 목소리가 강하다.",
        "이태원 참사 발생 100일째인 오는 5일 국회 차원의 추모제가 열린다. 여야가 공동으로 주최하는 이 행사는 국가기관에 의한 첫 공적 추모제에 해당한다.\n\n3일 이탄희 더불어민주당 의원실에 따르면, 이태원 참사 국회 추모제는 5일 오전 10시 국회 의원회관 대회의실에서 개최된다. 국회 이태원 참사 국정조사특별위원회가 주최하고, 국회 연구단체인 국회 생명안전포럼이 주관한다. 국회의원 등 국회 관계자들을 비롯해 유가족과 생존 피해자, 이태원 지역 상인, 목격자, 구조자도 참석할 예정이다.\n\n추모제에서 김진표 국회의장, 박홍근 민주당 원내대표, 주호영 국민의힘 원내대표, 이은주 정의당 원내대표, 용혜인 기본소득당 대표의 추모사와 함께 국회 차원의 다짐이 낭독될 계획이다. 생존 피해자와 최초 신고자의 증언과 유가족 발언, 종교계 추모 의례, 세월호 4·16 합창단 추모 공연도 진행된다.",
        ]
    
    date = [
        "2023-02-03",
        "2023-02-03",
        "2023-02-03",
        "2023-02-03",
        "2023-02-03",
        "2023-02-03",
        ]
    
    initial_comment = "나는 이 포스터를 좋아한다."
    
    
    for i in range(6):
        info = {
            "_id": ids[i],
            "post_title": post_title[i],
            "post_text": post_text[i],
            "time": date[i],
            "comments": [{"comment": initial_comment, "date": "2023-02-05", "fixed": ""}]
        }
        
        mongo.insert_one("poster", "post_info", info)
        
    return


def initial_post_for_test(mongo):
    post_info(mongo)
    return
    
def slicing_test(mongo):
    db = mongo.client["poster"]
    collection = db["post_info"]
    
    out = collection.update_one({"_id": 5}, { "$push" : {"comments": "fuck"}})
    
    out = collection.find({"_id": 5}, {"comments": {"$slice": -3}})
    for item in out:print(item)
    return

if __name__ == "__main__":
    mongo = MongoDB()
    
    #db = mongo.insert_one("hate_data", "labeled_count", {"date": '2023-02-04', "count": 0})
    #init_reward_database(mongo)
    #insert_reward_data(mongo)
    #mongo.drop_collection("hate_data", "check_count")
    #mongo.insert_one("hate_data", "check_count", {"id": -1, "count": 239})
    #fix_reward(mongo)
    #reward_data_extraction(mongo)
    
    #slicing_test(mongo)
    
    initial_post_for_test(mongo)
    mongo.insert_one("poster", "post_info", {"name": "counter", "count": 6})