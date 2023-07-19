import os
import torch

from app.neural_model.base_model import BaseModel
from config import settings


class YOLOV5(BaseModel):
    name_model = 'yolov5'

    def get_model(self, id_model):
        base_path = '{}/train_model/weights/best.pt'.format(id_model)
        model_weight = os.path.join(settings.MODEL_FOLDER, base_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.hub.load('app/neural_model/yolov5', 'custom', path=model_weight, source='local', device=device)
        return model

    def predict(self, id_model, input, model_cache):
        model_cache.eval()
        names = model_cache.module.names if hasattr(model_cache, 'module') else model_cache.names
        results = model_cache(input, size=int(640)).xyxy
        output = []
        for result in results:
            for r in result:
                output.append({'xmin': int(r[0]),
                               'ymin': int(r[1]),
                               'xmax': int(r[2]),
                               'ymax': int(r[3]),
                               'confidence': round(float(r[4]), 3),
                               'polygons': [[]],
                               'label': names[int(r[5])],
                               'label_index': int(r[5])
                               })
        return output
