from tensorflow.keras.layers import *
from jiwer import wer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import tensorflow as tf
from ctc_loss import CTC_loss

frame_step = 256
ctc = CTC_loss(frame_step)


def build_model(output_dim, custom_model, preprocess_model, mfcc=False, calc=None):

    input_audios = Input(name='the_input', shape=(None,))
    pre = preprocess_model(input_audios)
    pre = tf.squeeze(pre, [3])

    y_pred = custom_model(pre)
    model = Model(inputs=input_audios, outputs=y_pred, name="model_builder")
    model.output_length = calc

    return model


def build_model2(output_dim, cnn_model, custom_model, preprocess_model, mfcc=False, calc=None):

    input_audios = Input(name='the_input', shape=(None,))
    pre = preprocess_model(input_audios)
    pre = tf.squeeze(pre, [3])

    cnn_output = cnn_model(pre)

    y_pred = custom_model(cnn_output)
    model = Model(inputs=input_audios, outputs=y_pred, name="model_builder")
    model.output_length = calc

    return model


def train(model_builder,
          data_gen,
          batch_size=32,
          epochs=20,
          verbose=1,
          save_path="../models/model.h5",
          optimizer=SGD(learning_rate=0.01, decay=1e-6,
                        momentum=0.9, nesterov=True, clipnorm=5),
          ):

    model = ctc.add_ctc_loss(model_builder)

    checkpointer = ModelCheckpoint(filepath=save_path, verbose=0)
    model.compile(loss={'ctc': lambda y_true,
                  y_pred: y_pred}, optimizer=optimizer)
    print(model.summary())

    hist = model.fit_generator(generator=data_gen,
                               callbacks=[checkpointer],

                               epochs=epochs,
                               verbose=verbose,
                               use_multiprocessing=False)
    return model


def predict(model, audio, tokenizer, int_to_char, actual=None):

    pred_audios = tf.convert_to_tensor([audio])

    y_pred = model.predict(pred_audios)

    input_shape = tf.keras.backend.shape(y_pred)
    input_length = tf.ones(
        shape=input_shape[0]) * tf.keras.backend.cast(input_shape[1], 'float32')
    prediction = tf.keras.backend.ctc_decode(
        y_pred, input_length, greedy=False)[0][0]

    pred = K.eval(prediction).flatten().tolist()
    pred = [i for i in pred if i != -1]

    predicted_text = tokenizer.decode_text(pred, int_to_char)

    error = None
    if actual != None:
        error = wer(actual, predicted_text)

    return predicted_text, error
