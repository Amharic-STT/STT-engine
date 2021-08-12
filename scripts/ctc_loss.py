import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K


class CTC_loss:

    def __init__(self, hop_size):
        self.hop_size = hop_size

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def input_lengths_lambda_func(self, args):
        hop_size = self.hop_size
        input_length = args
        return tf.cast(tf.math.ceil(input_length/hop_size)-1, dtype="float32")

    def add_ctc_loss(self, model_builder):
        the_labels = Input(name='the_labels',
                           shape=(None,), dtype='float32')
        input_lengths = Input(name='input_length',
                              shape=(1,), dtype='float32')
        label_lengths = Input(name='label_length',
                              shape=(1,), dtype='float32')

        input_lengths2 = Lambda(self.input_lengths_lambda_func)(input_lengths)

        if model_builder.output_length:
            output_lengths = Lambda(
                model_builder.output_length)(input_lengths2)
        else:
            output_lengths = input_lengths2

        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [model_builder.output, the_labels, output_lengths, label_lengths])
        model = Model(inputs=[model_builder.input, the_labels,
                      input_lengths, label_lengths],  outputs=loss_out)
        return model
