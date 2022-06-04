import pickle
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from src.augmentation.augment import _load_characters
from src.utils.logger import logger
from sklearn.preprocessing import LabelEncoder
from src.utils.paths import CHARACTER_TRAIN_AUGMENTED_PATH, MODEL_PATH


class Classifier:
    def __init__(self, train: bool, model_filename: str, debug: bool = False) -> None:
        self.debug = debug
        self.model_path = f"{MODEL_PATH}/{model_filename}"
        self.label_encoder_path = f"{MODEL_PATH}/{model_filename}_encoder.pkl"
        self.label_encoder = LabelEncoder()

        if not train:
            logger.info(f"Loading existing model: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)

            with open(self.label_encoder_path, "rb") as inp:
                self.label_encoder = pickle.load(inp)
        else:
            logger.info(f"Training model and saving in {model_filename}")
            self.model = keras.models.Sequential(
                [
                    keras.layers.Conv2D(
                        6, 5, activation="tanh", input_shape=(32, 32, 1)
                    ),
                    keras.layers.AveragePooling2D(2),
                    keras.layers.Activation("sigmoid"),
                    keras.layers.Conv2D(16, 5, activation="tanh"),
                    keras.layers.AveragePooling2D(2),
                    keras.layers.Activation("sigmoid"),
                    keras.layers.Conv2D(120, 5, activation="tanh"),
                    keras.layers.Flatten(),
                    keras.layers.Dense(84, activation="tanh"),
                    keras.layers.Dense(27, activation="softmax"),
                ]
            )
            logger.info("Model architecture")

            self.model.compile(
                optimizer=tf.optimizers.Adam(),
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.metrics.SparseCategoricalAccuracy()],
            )

            self._train()

    def predict_batch(self, inp: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        prediction = self.model.predict(
            inp,
            batch_size=None,
            verbose="auto",
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )

        return prediction

    def decode_proba_batch(
        self, proba: npt.NDArray[np.float32]
    ) -> List[Tuple[str, float]]:

        if len(shape := proba.shape) < 2 or shape[1] != (
            num_classes := len(self.label_encoder.classes_)
        ):
            raise ValueError(f"Expect probabilities of shape (N, {num_classes}")

        max_label_encoded = np.argmax(proba, axis=1)
        max_proba = np.max(proba, axis=1).tolist()
        label = self.label_encoder.inverse_transform(max_label_encoded).tolist()

        return list(zip(label, max_proba))

    def _train(self) -> None:
        logger.info(
            f"Loading augmented characters from {CHARACTER_TRAIN_AUGMENTED_PATH}"
        )
        characters = _load_characters(CHARACTER_TRAIN_AUGMENTED_PATH, extension="png")
        characters_df = pd.DataFrame(characters, columns=["label", "filename", "image"])
        characters_df["label_encoded"] = self.label_encoder.fit_transform(
            characters_df["label"]
        )

        x_train, x_test, y_train, y_test = train_test_split(
            np.stack(characters_df["image"].to_list(), axis=0),
            np.stack(characters_df["label_encoded"].to_list(), axis=0),
            test_size=0.2,
            random_state=42,
            stratify=characters_df["label_encoded"].to_list(),
        )

        self.model.fit(
            x=x_train,
            y=y_train,
            epochs=100,
            batch_size=25,
            validation_data=(x_test, y_test),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, verbose=1, min_delta=1e-2
                )
            ],
            validation_split=0.2,
        )

        with open(self.label_encoder_path, "wb") as out:
            pickle.dump(self.label_encoder, out)

        self.model.save(self.model_path)
