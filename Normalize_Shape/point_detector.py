import os
import pytz
import datetime
import numpy as np
from typing import Tuple
from tensorflow import keras
from collections import namedtuple

from datagenerator import DataGenerator

# Data structs
DataGeneratorInputs = namedtuple(
    "DataGeneratorInputs", ["image_path", "points_xml_path"])


class PointDetector:
    """
    @brief Network to detect points on ceramic shape profiles.
           Network architecture based on Simple Baselines for Human Pose Estimation and Traking
           from Bin Xia et al.
    @param input_shape shape of input data (height, width, channels). Should be a RGB image because resnet is trained
            on RGB dat. Only used for training. Prediction can be done with arbitrarly width and height
    @param batch_size batch size
    @simga standard deviation of Gaussian heatmap
    @nr_keypoints Number of keypoints
    """

    def __init__(
        self,
        input_shape: Tuple,
        batch_size: int = 16,
        sigma: int = 3,
        nr_keypoints: int = 6,
    ):

        assert len(input_shape) == 3 and input_shape[2] == 3,\
            "Wrong input shape. Use 3 channel rgb data"

        self._input_shape = input_shape
        self._nr_keypoints = nr_keypoints
        self._sigma = sigma
        self._batch_size = batch_size
        self._save_dir = None
        self._data_generator = None
        self._val_generator = None

        # build model
        self.model = self._build_model()
        self._compile_model()

    def _build_model(self):
        """
        @brief build detection model with pretrained ResNet50 backbone.
        @return keras Model
        """
        base_model = keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(None, None, 3)
        )

        input_ = base_model.input
        x = base_model.output

        x = self.build_upsampling_network(
            x,
            (256, 256, 256),
            ("stage0", "stage1", "stage2"),
        )
        x = keras.layers.Conv2D(
            self._nr_keypoints,
            kernel_size=1,
            strides=1,
            activation=None,
            use_bias=False,
            name="final_conv",
            padding="same",
        )(x)
        out_model = keras.models.Model(input_, x)

        # freeze basemodel layers
        for layer in out_model.layers:
            if layer.name == "stage0_convT":
                break
            layer.trainable = False

        return out_model

    @staticmethod
    def build_upsampling_network(x, filters: Tuple, names: Tuple) -> keras.layers.Layer:
        """
        @brief Build upsampling network where each stage consists of Conv2DTrans->BN->RelU
        @param filters number of filters
        @names layer name
        """
        assert len(filters) == len(names)
        for name, _filter in zip(names, filters):
            x = keras.layers.Conv2DTranspose(
                _filter,
                kernel_size=4,
                strides=(2, 2),
                activation=None,
                padding="same",
                name=f"{name}_convT",
            )(x)
            x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
            x = keras.layers.Activation("relu", name=f"{name}_activation")(x)

        return x

    def _compile_model(self, optimizer=keras.optimizers.Adam(1e-4)) -> None:
        """
        @brief compiles models
        @param optimizer keras optimizer to train network
        """
        self.model.compile(optimizer=optimizer, loss="mse",
                           metrics=["mae", "mse"])

    def fit(self,  image_data: DataGeneratorInputs,
            val_data: DataGeneratorInputs, epochs: int = 40, save_dir: str = "save_dir"):
        """
        @brief perform training of point detector and creates tensorboard logdir
        @param epoch number of epochs to train network
        @param image_data DataGeneratorInputs namedtuple with image path and path of points exported as CVAT xml
        @param val_data DataGeneratorInputs namedtuple with val image path and path of points exported as CVAT xml
        @save_dir directory to save trained model and training stats
        @return keras.callbacks.History object
        """
        self.setup_datagenerator(image_data, val_data)
        self.setup_savedir(save_dir)

        run_logdir = os.path.join(
            self._save_dir,
            datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime(
                "%Y%m%d-%H%M%S"
            ),
        )
        print(f'Tensorboard --logdir {run_logdir}')

        tensorboard_cb = keras.callbacks.TensorBoard(
            run_logdir, histogram_freq=1, update_freq='epoch')
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            os.path.join(self._save_dir, "best_model.h5"))
        history = self.model.fit_generator(
            self._data_generator,
            epochs=epochs,
            callbacks=[tensorboard_cb, checkpoint_cb],
            validation_data=self._val_generator,
        )

        self.model.save_weights(os.path.join(
            self._save_dir), f'final_trained_weigths_with_{epochs}_epochs.h5')
        return history

    def setup_savedir(self, save_dir: str):
        """
        @brief stup save dir and mkdir if not exists
        @save_dir directory to save trained model and training stats
        """
        self._save_dir = save_dir
        if not os.path.exists(os.path.join(os.curdir, self._save_dir)):
            os.makedirs(self._save_dir)

    def setup_datagenerator(self, image_data: DataGeneratorInputs, val_data: DataGeneratorInputs):
        """
        @brief setup data generators for training
        @param image_data DataGeneratorInputs namedtuple with image path and path of points exported as CVAT xml
        @param val_data DataGeneratorInputs namedtuple with val image path and path of points exported as CVAT xml
        """
        gen_arcs = {
            "image_size": self._input_shape[:2],
            "batch_size": self._batch_size,
            "sigma": self._sigma,
        }
        self._data_generator = DataGenerator(
            image_path=image_data.image_path, points_xml_path=image_data.points_xml_path, **gen_arcs)
        self._val_generator = DataGenerator(
            image_path=val_data.image_path, points_xml_path=val_data.points_xml_path, **gen_arcs)

        assert getattr(self._data_generator, 'nr_keypoints') == getattr(self._val_generator, 'nr_keypoints'),\
            "Validation Data has not same number of keypoint classes as Training Data"
        assert self._nr_keypoints == getattr(
            self._data_generator, 'nr_keypoints')

    def __repr__(self):
        return f"Point Detector network. Keypoints: {self._nr_keypoints}\
                input_shape = {self._input_shape}"

    def __str__(self):
        return self.__repr__()

    def summary(self):
        self.model.summary()

    def predict(self, data_generator: DataGenerator) -> np.ndarray:
        return self._model.predict_generator(data_generator)

    def predict_img(self, img: np.ndarray) -> np.ndarray:
        return self.model.predict(img[np.newaxis, ...]
                                  if len(img.shape) == 3 else img)

    def load_weights(self, modelpath: str):
        self.model.load_weights(modelpath)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
