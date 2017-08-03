# Arda Mavi

import os
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, concatenate

def save_model(model):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    model_json = model.to_json()
    with open("Data/Model/model.json", "w") as model_file:
        model_file.write(model_json)
    model.save_weights("Data/Model/weights.h5")
    print('Model and weights saved')
    return


def get_model():
    img_inputs = Input(shape=(700, 700, 3))
    lidar_inputs = Input(shape=(3,))

    conv_1 = Conv2D(64, (3,3), strides=(1,1))(img_inputs)
    act_1 = Activation('relu')(conv_1)

    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_1)

    conv_2 = Conv2D(64, (3,3), strides=(1,1))(pooling_1)
    act_2 = Activation('relu')(conv_2)

    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_2)

    conv_3 = Conv2D(128, (3,3), strides=(1,1))(pooling_2)
    act_3 = Activation('relu')(conv_3)

    pooling_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_3)

    flat_1 = Flatten()(pooling_3)

    fc = Dense(128)(flat_1)

    lidar_fc = Dense(360)(lidar_inputs)

    lidar_fc = Dense(10)(lidar_fc)

    concatenate_layer = concatenate([fc, lidar_fc])

    fc = Dense(128)(concatenate_layer)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)

    direction_fc = Dense(10)(fc)
    speed_fc = Dense(10)(fc)

    direction = Dense(1)(direction_fc)
    speed = Dense(1)(speed_fc)

    outputs = Activation('sigmoid')(fc)

    model = Model(inputs=[img_inputs, lidar_inputs], outputs=[direction, speed])

    model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

    print(model.summary())

    return model

if __name__ == '__main__':
    save_model(get_model())
