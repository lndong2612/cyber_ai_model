
class BaseModel():
    name_model = None

    def __init__(self) -> None:
        super().__init__()

    def get_name(self):
        return self.name_model

    def get_model(self, id_model):
        raise NotImplementedError

    def train_model(self, id_model):
        raise NotImplementedError

    def attack(self):
        raise NotImplementedError

    def predict(self, id_model, input, model_cache):
        raise NotImplementedError

    def plot_bbox_image(self, image, bboxes):
        raise NotImplementedError