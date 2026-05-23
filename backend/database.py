from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv() # load the env file

MONGO_URL = os.getenv('MONGO_URL')
DATABASE_NAME = os.getenv('DATABASE_NAME')

client = MongoClient(MONGO_URL)

db = client[DATABASE_NAME]