from random import seed
from typing import Optional
import tensorflow as tf
from tensorflow.keras import models, layers

from tensorflow import keras
from src.utils.logger import logger
import pickle
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


LABELS = [
    "Alef",
    "Ayin",
    "Bet",
    "Dalet",
    "Gimel",
    "He",
    "Het",
    "Kaf",
    "Kaf-final",
    "Lamed",
    "Mem",
    "Mem-medial",
    "Nun-final",
    "Nun-medial",
    "Pe",
    "Pe-final",
    "Qof",
    "Resh",
    "Samekh",
    "Shin",
    "Taw",
    "Tet",
    "Tsadi-final",
    "Tsadi-medial",
    "Waw",
    "Yod",
    "Zayin",
]


class Classifier:
    def __init__(
        self,
        predict_image_from_path: Optional[str] = None,
    ) -> None:

        self.data_path = "data/unpacked/characters"

        if predict_image_from_path:
            self.get_model()
            logger.info(
                f"Predicted label is {self.predict_from_path(predict_image_from_path)}"
            )

    def train_model(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = self._load_data()
        self.create_model()
        self._compile_model()
        self.fit_model()
        self._save_model()

    def get_model(self) -> None:

        try:
            self.model = models._load_model("model.h5")
            logger.info("Model loaded")
        except:
            logger.info("model not found, creating new model")
            self.train_model()

    def create_model(self) -> None:
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
        self.model = model

    def _compile_model(self) -> None:
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    def _callbacks(self) -> list:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, verbose=1, min_delta=1e-2
            )
        ]
        return callbacks

    def fit_model(self) -> None:
        self.model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=100,
            batch_size=25,
            validation_data=(self.x_test, self.y_test),
            # callbacks=self._callbacks(),
            validation_split=0.2,
        )

    def _save_model(self) -> None:
        self.model.save("model.h5")

    def load_model(self) -> None:
        try:
            self.model = models.load_model("model.h5")
            logger.info("Model loaded")
        except:
            logger.info(
                "model not found, creating/train a new model first with --train"
            )

    def _load_data(self) -> tuple:

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
        print(self._inverse_transform(pred))
        return pred

    def predict_from_path(self, path: str) -> str:
        img = cv2.imread(path, 0)
        return self.predict(img)

    def _inverse_transform(self, pred: np.ndarray) -> str:
        le = preprocessing.LabelEncoder()
        le.fit(LABELS)
        return le.inverse_transform(np.argmax(pred, axis=1))[0]
        return le.inverse_transform(pred)

    def _encode_labels(self, labels: list) -> np.ndarray:
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        return le.transform(labels)

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

        lb_arr = self._encode_labels(lb_arr)

        # dump the data to pickle
        with open("data.pkl", "wb") as f:
            pickle.dump(im_arr, f)
            pickle.dump(lb_arr, f)

        return im_arr, lb_arr
