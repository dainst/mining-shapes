import os
import pytz
import datetime
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
    @param image_data DataGeneratorInputs namedtuple with image path and path of points exported as CVAT xml
    @param val_data DataGeneratorInputs namedtuple with val image path and path of points exported as CVAT xml
    @param input_shape shape of input data (height, width, channels). Should be a RGB image because resnet is trained 
            on RGB dat
    @param batch_size batch size
    @simga standard deviation of Gaussian heatmap 
    @save_dir directory to save trained model and training stats
    """

    def __init__(
        self,
        image_data: DataGeneratorInputs,
        val_data: DataGeneratorInputs,
        input_shape: Tuple,
        batch_size: int,
        sigma: int = 2,
        save_dir: str = "save_dir",
    ):

        assert len(input_shape) == 3 and input_shape[2] == 3,\
            "Wrong input shape. Use 3 channel rgb data"

        self._input_shape = input_shape

        gen_arcs = {
            "image_size": self._input_shape[:2],
            "batch_size": batch_size,
            "sigma": sigma,
        }
        self._data_generator = DataGenerator(
            image_path=image_data.image_path, points_xml_path=image_data.points_xml_path, **gen_arcs)
        self._val_generator = DataGenerator(
            image_path=val_data.image_path, points_xml_path=val_data.points_xml_path, **gen_arcs)

        assert getattr(self._data_generator, 'nr_keypoints') == getattr(self._val_generator, 'nr_keypoints'),\
            "Validation Data has not same number of keypoint classes as Training Data"
        self._nr_keypoints = getattr(self._data_generator, 'nr_keypoints')
        self.model = self._build_model()
        self._compile_model()

        # set up save directory
        self._save_dir = save_dir
        if not os.path.exists(os.path.join(os.curdir, self._save_dir)):
            os.makedirs(self._save_dir)

    def _build_model(self):
        """
        @brief build detection model with pretrained ResNet50 backbone.
        @return keras Model
        """
        base_model = keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=self._input_shape
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

        #freeze basemodel layers
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

    def fit(self, epochs: int = 40):
        """
        @brief perform training of point detector and creates tensorboard logdir
        @param epoch number of epochs to train network
        @return keras.callbacks.History object
        """
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

    def __repr__(self):
        return f"Point Detector network. Keypoints: {self._nr_keypoints}\
                input_shape = {self._input_shape}"

    def __str__(self):
        return self.__repr__()

    def summary(self):
        self.model.summary()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self,value):
        self._model = value

