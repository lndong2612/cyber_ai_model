from mongoengine import connect
from config import settings
from flask_executor import Executor
from .model_db import Model
from .dataset_db import Dataset
from .image_db import Image
from .label_db import Label

try:
    executor = Executor()
    if settings.REPLICA:
        print("connect_db")
        connect(settings.MONGODB_DB, host=settings.MONGO_DATABASE_URI)
except:
    print("error connect db")
