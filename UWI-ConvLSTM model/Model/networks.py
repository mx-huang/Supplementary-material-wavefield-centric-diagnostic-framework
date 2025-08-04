from keras.layers import AveragePooling3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D

def convlstm_model(input_shape):
    model = Sequential([
        ConvLSTM2D(filters=12, kernel_size=3, padding='same', return_sequences=True,input_shape=input_shape),
        BatchNormalization(),
        ConvLSTM2D(filters=6, kernel_size=3, padding='same', return_sequences=True),
        BatchNormalization(),
        ConvLSTM2D(filters=12, kernel_size=3, padding='same', return_sequences=False),
        BatchNormalization(),
        Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same',data_format='channels_last')
    ])
    return model
