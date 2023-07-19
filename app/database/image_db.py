from mongoengine import Document, StringField, ListField, DictField, IntField, BooleanField


class Image(Document):
    """
    name images: tên ảnh
    image_uri : đươgn dẫn tới ảnh
    id_dataset : id dataset lưu ảnh
    label : nhãn ảnh có dạng path_image xmin, ymin, xmax, ymax, class_id xmin, ymin, xmax, ymax, class_id
    """
    meta = {'collection': 'Image', 'strict': False}

    img_type = StringField()
    is_deleted = BooleanField(default=False)
    labels = ListField()
    img_name = StringField()
    img_originalname = StringField()
    img_path = StringField()

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
