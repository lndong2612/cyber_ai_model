import os
import tensorflow as tf
from app.neural_model.attention_ocr.model import Model
from app.neural_model.attention_ocr.metrics import loss_function, display_validate
import time
from app.neural_model.attention_ocr.config_model import *
from app.neural_model.attention_ocr.data_generator import Generator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model = Model(decode_units=decode_units, vocab_size=vocab_size, image_height=image_height, image_width=image_width,
              finetune=False)


# @tf.function
def train_step(images, word_target):  # word_target shape(bs, max_txt_length, vocab_size)
    loss = 0

    hidden = tf.zeros((BATCH_SIZE, decode_units))
    word_one_hot = word_target[:, 0, :]  # corresponding the word 'START'
    with tf.GradientTape() as tape:
        # Teacher forcing - feeding the target as the next input
        for i in range(1, word_target.shape[1]):
            y_pred, hidden = model(word_one_hot, hidden, images)
            word_one_hot = word_target[:, i, :]

            loss += loss_function(word_target[:, i, :], y_pred)

    batch_loss = loss / int(word_target.shape[1])
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


if __name__ == '__main__':
    time_tuple = time.localtime()
    model_id = time.strftime('%d%m%Y_%H%M%S',time_tuple)

    if not os.path.exists('models'):
        os.mkdir('models')
    os.mkdir('models/{}'.format(model_id))

    generator_training = Generator(folder_image='train', 
                                   folder_label='train.txt')
    
    generator_valid = Generator(folder_image='val',
                                folder_label='val.txt')
    
    # # if args['finetune']:
    # #     model.load_weights('model.h5')

    print('Number data train: ', len(generator_training.examples))
    print('Number data val: ', len(generator_valid.examples))

    print('\nStart training ...\n')

    step_per_epoch_training = len(generator_training.examples) // BATCH_SIZE

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        for i in range(step_per_epoch_training):
            imgs, target = next(generator_training.examples_generator())
            total_loss += train_step(imgs, target)

        if epoch % 5 == 0 and epoch != 0:
            display_validate(generator_valid, model)
            model.save_weights(f'models/{model_id}/model_epoch{epoch}')
        else:
            pass

        print('Epoch {}/{} Loss {:.6f}'.format(epoch + 1, EPOCHS, total_loss / step_per_epoch_training))
        print('Time taken for 1 epoch {:.6f} sec\n'.format(time.time() - start))
