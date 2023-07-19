from app.neural_model.attention_ocr.model import Model
from app.neural_model.attention_ocr.config_model import *

# call aocr trained model from path with id_model
def load_aocr_model(id_model):
    model = Model(decode_units=decode_units,
                vocab_size=vocab_size,
                image_height=image_height,
                image_width=image_width,
                finetune=False,
                visual_attention=True)
    model.load_weights(id_model)
    return model