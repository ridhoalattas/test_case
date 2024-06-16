import pymongo
from datetime import datetime

DB_HOST = "localhost"
DB_PORT = "27017"
DB_NAME = "E-FISHERY"

# if os.getenv("DB_MONGO_USERNAME") is None or os.getenv("DB_MONGO_PASSWORD") is None:
connection_string = "mongodb://" + DB_HOST + ":" + DB_PORT + "/"

db = None
try:
    client = pymongo.MongoClient(connection_string)
    client.server_info()
except pymongo.errors.ServerSelectionTimeoutError as e:
    print("[ERROR]", e)
db = client[DB_NAME]
if db is not None : 
    mongo_fishery = db['original']
else : 
    print('[INFO] Gagal terhubung ke database')

def insert_data(data:dict):
    
    data['created_at'] = datetime.now()
    data['deleted_at'] = None
    
    tagging = mongo_fishery.insert_one(data)
    return True
