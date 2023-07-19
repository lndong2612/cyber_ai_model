import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/model"
    DEBUG = False
    PROJECT_NAME: str = "Cybersecurity Model API"
    SECRET_KEY: str = "cyber-security-secret-key"
    MONGO_DATABASE_URI = "mongodb://camnv:thinklAb202x@192.168.1.12:2700,10.192.168.1.14:2700,192.168.1.16:2700/cyber_security?replicaset=replicasetC1&authSource=cyber_security"
    MONGODB_DB: str = "cybersecurity"
    REPLICA: bool = True

    RESOURCES: str = "public/files"
    RESOURCES_NODE: str = "public/files"  # Đường dẫn lưu ảnh bên mô hình backend

    IMAGE_FOLDER: str = os.path.join(RESOURCES, 'uploads', 'images')
    MODEL_FOLDER: str = os.path.join(RESOURCES, 'models')
    MODEL_WEIGHT_INIT: str = os.path.join(RESOURCES, 'weight_init')
    TEST_FOLDER = os.path.join(RESOURCES, 'test_image')

    HOST: str = '127.0.0.1'
    PORT: int = 4002

    JWT_ALGORITHM: str = "HS256"

    THRESHOLD_FACE = 0.5
    MAX_MODEL = 2

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()
settings.IMAGE_FOLDER: str = os.path.join(settings.RESOURCES, 'uploads', 'images')
settings.MODEL_FOLDER: str = os.path.join(settings.RESOURCES, 'models')
settings.MODEL_WEIGHT_INIT: str = os.path.join(settings.RESOURCES, 'weight_init')
settings.TEST_FOLDER = os.path.join(settings.RESOURCES, 'test_image')

# print(os.environ.get(
#     "BENTOML_HOME", os.path.join(os.path.expanduser("~"), "bentoml")
# ))
