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
    img_inputs = Input(shape=(500, 500, 1))
    lidar_inputs = Input(shape=(3,))

    conv_1 = Conv2D(32, (4,4), strides=(2,2))(img_inputs)

    conv_2 = Conv2D(32, (4,4), strides=(2,2))(conv_1)

    conv_3 = Conv2D(32, (3,3), strides=(1,1))(conv_2)
    act_3 = Activation('relu')(conv_3)

    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_3)

    flat_1 = Flatten()(pooling_1)

    fc = Dense(32)(flat_1)

    lidar_fc = Dense(32)(lidar_inputs)

    concatenate_layer = concatenate([fc, lidar_fc])

    fc = Dense(10)(concatenate_layer)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)

    outputs = Dense(2)(fc)

    outputs = Activation('sigmoid')(outputs)


    model = Model(inputs=[img_inputs, lidar_inputs], outputs=[outputs])

    model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

    print(model.summary())

    return model

if __name__ == '__main__':
    save_model(get_model())
