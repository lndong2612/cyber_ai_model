from mongoengine import Document, StringField, ListField, DictField, IntField, ListField, BooleanField


class Dataset(Document):
    meta = {'collection': 'Dataset', 'strict': False}

    dataset_type = StringField(required=True)
    images = ListField(StringField())
    dataset_name = StringField(required=True)
    label_cate = ListField()
    dataset_status = StringField()
    description = StringField()
    role_user = StringField()
    is_deleted = BooleanField(default=False)
    createdAt = StringField()
    updatedAt = StringField()

    @classmethod
    def get(cls, id_dataset):
        try:
            return cls.objects.get(id=id_dataset)
        except cls.DoesNotExist:
            return None
