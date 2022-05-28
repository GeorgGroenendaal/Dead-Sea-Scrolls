import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics
import matplotlib.pyplot as plt
from tensorflow import keras
from src.utils.logger import logger
import pickle
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class Classifier:
    def __init__(
        self, data_path: str, predicit: bool = False, predict_image: str = None
    ) -> None:
        self.data_path = data_path
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_data()
        self.get_model()

    def get_model(self) -> None:

        try:
            self.model = models.load_model("model.h5")
            logger.info("Model loaded")
        except:
            logger.error("model not found")
            self.model = self.create_model()
            self.compile_model()
            self.fit_model()
            self.save_model()
            logger.info("model saved")

    def create_model(self):
        model = models.Sequential(
            [
                layers.Conv2D(6, 5, activation="tanh", input_shape=(32, 32, 1)),
                layers.AveragePooling2D(2),
                layers.Activation("sigmoid"),
                layers.Conv2D(16, 5, activation="tanh"),
                layers.AveragePooling2D(2),
                layers.Activation("sigmoid"),
                layers.Conv2D(120, 5, activation="tanh"),
                layers.Flatten(),
                layers.Dense(84, activation="tanh"),
                layers.Dense(27, activation="softmax"),
            ]
        )
        return model

    def compile_model(self) -> None:
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    def _callbacks(self) -> None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, verbose=1, min_delta=1e-2
            )
        ]
        self.callbacks = callbacks

    def fit_model(self) -> None:
        self.model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=60,
            batch_size=64,
            validation_data=(self.x_test, self.y_test),
            callbacks=self._callbacks(),
            validation_split=0.2,
        )

    def save_model(self) -> None:
        self.model.save("model.h5")

    def load_model(self) -> None:
        self.model = models.load_model("model.h5")

    def load_data(self):

        try:
            with open("data.pkl", "rb") as f:
                im_arr = pickle.load(f)
                new_labels = pickle.load(f)
            logger.info("prepocessed data loaded")
        except:
            logger.info("preprocess data not found, starting preprocessing...")
            im_arr, new_labels = self.preprocess_data()

        x_train, x_test, y_train, y_test = train_test_split(
            im_arr, new_labels, test_size=0.2, random_state=42, stratify=new_labels
        )

        # expand dims for required input shape
        x_train = tf.expand_dims(x_train, axis=3, name=None) / 255
        x_test = tf.expand_dims(x_test, axis=3, name=None) / 255

        return x_train, x_test, y_train, y_test

    def predict(self, img):
        img = cv2.resize(img, (32, 32))
        img = img[np.newaxis, :, :]
        img = img.astype("float32") / 255
        pred = self.model.predict(img)
        return pred

    def predict_from_path(self, path: str) -> str:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (32, 32))
        img = img[np.newaxis, :, :]
        img = img.astype("float32") / 255
        pred = self.model.predict(img)
        return self._revert_label(pred)

    def _revert_label(self, pred: np.ndarray) -> str:
        le = preprocessing.LabelEncoder()
        le.fit(self.y_train)
        return le.inverse_transform(pred)

    def preprocess_data(self):
        characters = os.listdir(self.data_path)

        im_arr = []
        lb_arr = []
        for i, character in enumerate(characters):
            path = os.path.join(self.data_path, character)

            for d in sorted(os.listdir(path)):
                img = cv2.imread(os.path.join(path, d), 0)
                img = cv2.resize(img, (32, 32))

                temp_arr = img
                temp_arr = temp_arr[np.newaxis, :, :]
                if len(im_arr) == 0:
                    im_arr = temp_arr
                else:
                    im_arr = np.concatenate((im_arr, temp_arr), axis=0)
                lb_arr.append(character)

        lb_arr = np.array(lb_arr)
        le = preprocessing.LabelEncoder()
        le.fit(lb_arr)
        new_labels = le.transform(lb_arr)

        # dump the data to pickle
        with open("data.pkl", "wb") as f:
            pickle.dump(im_arr, f)
            pickle.dump(new_labels, f)

        return im_arr, new_labels
