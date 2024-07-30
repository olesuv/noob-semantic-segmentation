import numpy as np

from tensorflow.keras import models


class TrainModel:
    def __init__(self, X_train: np.array, y_train: np.array, model: models.Model):
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        print(self.model)

    def train(self, epochs=30, batch_size=16, validation_split=0.1):
        self.model = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                    validation_split=validation_split)

    def __get__(self):
        return self.model

    def test_model(self, X_test: np.array, y_test: np.array):
        return self.model.evaluate(X_test, y_test)

    def model_predict(self, X_test: np.array):
        return self.model.predict(X_test)
