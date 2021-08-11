import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras import backend as K


def BidirectionalRNN(input_dim, batch_size, sample_rate=22000,
                     rnn_layers=2, units=400, drop_out=0.5, act='tanh', output_dim=224):

    input_data = Input(name='the_input', shape=(
        None, input_dim), batch_size=batch_size)

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
