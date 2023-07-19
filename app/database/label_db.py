from mongoengine import Document, StringField


class Label(Document):
    meta = {'collection': 'Label', 'strict': False}

    labelName = StringField()
    datasetId = StringField()

    @classmethod
    def get(cls, id_label):
        try:
            return cls.objects.get(id=id_label)
        except cls.DoesNotExist:
            return None
