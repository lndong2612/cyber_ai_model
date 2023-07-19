from cachetools import Cache
from app.neural_model.list_model import all_model

class ModelRepository():

    def get_model(self, id_model, model_arch):
        model_architecture = all_model.get(model_arch)
        if model_architecture:
            model = model_architecture().get_model(id_model)
            return model


class CachedModelRepository():
    def __init__(self, cache: Cache):
        self.cache = cache

    def get_model(self, id_model, model_arch, model_repository):
        if id_model not in self.cache:
            self.cache[id_model] = model_repository.get_model(id_model,model_arch)
        return self.cache[id_model]

    def delete_model(self, id_model):
        if id_model in self.cache:
            self.cache.pop(id_model)
