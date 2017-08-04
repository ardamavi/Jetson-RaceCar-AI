# Arda Mavi

import os
import numpy as np
from get_data import get_data
from get_model import get_model, save_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

def train_model(model, X_1, X_2, Y):

    batch_size = 1
    epochs = 10

    checkpoints = []
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')
    checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

    model.fit([X_1, X_2], Y, batch_size=batch_size, epochs=epochs, validation_data=([X_1, X_2], Y), shuffle=True, callbacks=checkpoints)

    return model

def main():
    X_1, X_2, Y = get_data()
    model = train_model(get_model(), X_1, X_2, Y)
    save_model(model)
    return model

if __name__ == '__main__':
    main()
