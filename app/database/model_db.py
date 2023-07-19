import logging
from mongoengine import Document, StringField, IntField, DictField, BooleanField, FloatField, ListField

logger = logging.getLogger(__name__)


class Model(Document):
    """
    model_status: trạng thái model init- khởi tạo, training - đào tạo, trained- đã chạy xong, deploy_active - triển khai
    deploy_inactive: không triển khai, failed - Lỗi không đào tạo được, testing - đang kiểm tra
    """
    meta = {'collection': 'Model', 'strict': False}
    model_name = StringField(required=True)
    model_status = StringField(required=True)
    model_type = StringField(required=True)
    architecture = StringField(required=True)
    labels = ListField()
    dataset = StringField()
    user = StringField()
    learning_rate = FloatField()
    batch_size = IntField()
    split_train = FloatField()
    epochs = IntField()
    description = StringField()
    config = DictField()
    evalution = DictField()
    log = DictField()
    is_deleted = BooleanField()
    createdAt = StringField()
    updatedAt = StringField()

    @classmethod
    def all(cls):
        return cls.objects.all()

    @classmethod
    def get(cls, id_model):
        try:
            model = cls.objects.get(id=id_model)
            return model
        except cls.DoesNotExist:
            return None

    @classmethod
    def update(cls, id_model, data_update):
        try:
            if cls.is_exist_model(id_model):
                model = cls.objects(id=id_model).update(
                    __raw__={'$set': data_update})
                return cls.get(id_model)
            else:
                logger.error("Mô hình không tồn tại!")
        except Exception as e:
            logger.error("Cập nhật mô hình không thành công!" + str(e))
        return None

    @classmethod
    def is_exist_model(cls, id_model):
        return cls.objects(id=id_model).first()
