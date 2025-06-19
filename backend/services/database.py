import os 
from pymongo import MongoClient
from dotenv import load_dotenv


load_dotenv()
client=MongoClient(os.getenv("MONGO_URl"))
db=client[os.getenv("MONGO_DB_NAME")]

def get_db():
    return db