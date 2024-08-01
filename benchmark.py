import os
import numpy as np
import matplotlib.pyplot as plt

from utils import save_results_img
from tensorflow.keras import models


class ModelBenchmark:
    def __init__(self, trained_model: models.Model, train_hist, X_test: np.array, y_test: np.array) -> None:
        self.model = trained_model
        self.train_hist = train_hist
        self.X_test = X_test
        self.y_test = y_test

    def test_model(self):
        return self.model.evaluate(self.X_test, self.y_test)

    def benchmarks_dir(self):
        if os.path.exists('./benchmarks'):
            return
        os.makedirs('./benchmarks')

    def plot_accuracy_val_accuracy(self):
        self.benchmarks_dir()

        plt.plot(self.train_hist.history['accuracy'])
        plt.plot(self.train_hist.history['val_accuracy'])
        plt.title('accuracy vs validation accuracy')
        plt.legend(["accuracy", "val_accuracy"], loc="lower right")
        plt.savefig('./benchmarks/acc_val_acc.png')
        plt.show()

    def plot_dice_score_loss(self):
        self.benchmarks_dir()

        plt.plot(self.train_hist.history['dice_score'])
        plt.plot(self.train_hist.history['loss'])
        plt.title('dcie score vs loss')
        plt.legend(["dice_score", "loss"], loc="lower right")
        plt.savefig('./benchmarks/dice_score_loss.png')
        plt.show()

    def plot_example_results(self):
        predicted_imgs = self.model.predict(self.X_test)
        for i in range(5):
            save_results_img(
                self.X_test[i], predicted_imgs[i], self.y_test[i], i)
