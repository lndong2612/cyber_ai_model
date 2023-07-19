from mongoengine import Document, StringField, ListField, DictField, IntField, BooleanField


class Image(Document):
    meta = {'collection': 'ImageNoise', 'strict': False}

    img_type = StringField()
    labels = ListField()
    img_name = StringField()
    img_originalname = StringField()
    img_path = StringField()
    is_deleted = BooleanField(default=False)

    @classmethod
    def get_from_dataset(cls, id_dataset):
        pass

    @classmethod
    def all(cls):
        pass

    @classmethod
    def get(cls, id_image):
        try:
            return cls.objects.get(id=id_image)
        except cls.DoesNotExist:
            return None
