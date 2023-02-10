import logging
import asyncio, time
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient


class MongoDB():
    def __init__(self):
        self.client = AsyncIOMotorClient(host="localhost", port=27017)
        self.logger = logging.getLogger()
        
    async def insert_one(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = await collection.insert_one(data)
        except:
            print("Insert one function have errors")
    
    
    async def insert_many(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]

        try:
            output = await collection.insert_many(data)
        except:
            print("Insert many function have errors")
           
            
    async def delete_one(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]

        try:
            output = await collection.delete_one(data)
        except:
            print("Delete one function have errors")
    
    
    async def delete_many(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]

        try:
            output = await collection.delete_many(data)
        except:
            print("Delete many function have errors")
    
            
    async def update_one(self, db_name, collection_name, old_data, new_data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = await collection.update_one(old_data, {"$set": new_data})
        except:
            print("Update one function have errors")
            return None
    
    async def update_many(self, db_name, collection_name, old_data, new_data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = await collection.update_many(old_data, {"$set": new_data})
            return output
        except:
            print("Update many function have errors")
            return None
    
    async def find_one(self, db_name, collection_name, data, start_num=None, end_num=None):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = await collection.find_one(data)
            return output
        except:
            print("Find one function have errors")
            return None
    
    
    async def find_many(self, db_name, collection_name, data, sort_key=None, limit=None, length=None):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            if sort_key is not None:
                if limit == None:
                    output = await collection.find(data).sort(sort_key).to_list(length)
                else:
                    output = await collection.find(data).sort(sort_key, -1).limit(limit).to_list(limit)
            else:
                if limit == None:
                    output = await collection.find(data).to_list(length)
                else:
                    output = await collection.find(data).limit(limit).to_list(limit)
                    
            return output
        
        except Exception as e:
            print(e)
            print("Find many function have errors")
            return None
    
    async def push_data(self, db_name, collection_name, to_find, data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = await collection.update_one(to_find, {"$push": data})
        except:
            print("Push data function have errors")
    
        return None

    async def slicing(self, db_name, collection_name, to_find, start_num, end_num=None):
        db = self.client[db_name]
        collection = db[collection_name]
        
        if end_num == None:
            ## TODO: why can not use "await"
            try:
                output = await collection.find(to_find, {"comments": {"$slice": start_num}}).to_list(1)
            except:
                print("Slicing function have errors")
                return None
        else:
            try:
                output = await collection.find(to_find, {"comments": {"$slice": [start_num, end_num]}}).to_list(1)
            except:
                print("Slicing function have errors")
                return None
            
        return output
        
    
    async def count_documents(self, db_name, collection_name, data):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = await collection.count_documents(data)
            return output
        except:
            print("Count documents function have errors")
            return None
       
        
    async def drop_collection(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        
        try:
            output = await collection.drop()
        except:
            print("Drop collection function have errors")
    
    
    async def drop_database(self, db_name):
        try:
            await self.client.drop_database(db_name)
        except:
            print("Drop database function have errors")