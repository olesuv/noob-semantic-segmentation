import numpy as np

from tensorflow.keras import models


class TrainModel:
    def __init__(self, X_train: np.array, y_train: np.array, model: models.Model, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model

    def train(self, epochs=20, batch_size=16):
        return self.model.fit(self.X_train, self.y_train, epochs=epochs,
                              batch_size=batch_size, validation_data=(self.X_val, self.y_val))

    def __get__(self):
        return self.model
