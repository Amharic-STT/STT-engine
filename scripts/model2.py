from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow.keras.layers import *

from logspectrorgam import LogMelSpectrogram


def simple_rnn_model(input_dim, output_dim=224):

    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = GRU(output_dim, return_sequences=True,
                   implementation=2, name='rnn')(input_data)
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    model = Model(inputs=input_data, outputs=y_pred, name="simple_rnn_model")
    model.output_length = lambda x: x
    return model


def BidirectionalRNN2(input_dim, batch_size, sample_rate=8000,
                      rnn_layers=4, units=400, drop_out=0.5, act='tanh', output_dim=224):

    input_data = Input(name='the_input', shape=(
        None, input_dim))

    x = Bidirectional(LSTM(units,  activation=act,
                      return_sequences=True, implementation=2))(input_data)

    x = BatchNormalization()(x)
    x = Dropout(drop_out)(x)

    for i in range(rnn_layers - 2):
        x = Bidirectional(
            LSTM(units, activation=act, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(drop_out)(x)

    x = Bidirectional(LSTM(units,  activation=act,
                      return_sequences=True, implementation=2))(x)
    x = BatchNormalization()(x)
    x = Dropout(drop_out)(x)

    time_dense = TimeDistributed(Dense(output_dim))(x)

    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred, name="BidirectionalRNN")

    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
                  conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
                         return_sequences=True,  name='rnn')(bn_cnn)
    # TODO: Add batch normalization

    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    # model.output_length = lambda x: x

    return model


def CNN_net(n_mels):

    input_data = Input(name='the_input', shape=(
        None, n_mels, 1))

    y = Conv2D(128, (7, 7), padding='same')(input_data)  # was 32
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D((1, 2))(y)

    y = Conv2D(64, (5, 5), padding='same')(y)  # was 32
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D((1, 2))(y)

    y = Conv2D(64, (3, 3), padding='same')(y)  # was 32
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D((1, 2))(y)

    y = Reshape((-1, y.shape[-1] * y.shape[-2]))(y)

    model = Model(inputs=input_data, outputs=y, name="cnn")
    return model, model.output.shape


def preprocessin_model(sample_rate, fft_size, frame_step, n_mels, mfcc=False):

    input_data = Input(name='input', shape=(None,), dtype="float32")
    featLayer = LogMelSpectrogram(
        fft_size=fft_size,
        hop_size=frame_step,
        n_mels=n_mels,

        sample_rate=sample_rate,
        f_min=0.0,

        f_max=int(sample_rate / 2),
    )(input_data)

    x = BatchNormalization(axis=2)(featLayer)
    model = Model(inputs=input_data, outputs=x, name="preprocessin_model")

    return model
