"""
Skript to train shape segmentation model. Model is mainly best on the segmentation_models package. 
See https://github.com/qubvel/segmentation_models for more details.
"""

import argparse
import segmentation_models as sm
from datagenerator import DataGenerator
import datetime
import pytz
import keras
import os
import pickle

# pylint: disable=W0612


def train_model(_args):
    """
    @brief: Function to train segmentation model
    @param _args command line arguments
    """
    backbone = _args.backbone
    if _args.input_shape[0] == "None":
        image_size = None
    else:
        image_size = [int(i) for i in _args.input_shape]
    epochs = int(_args.epochs)
    batch_size = int(_args.batch_size)
    save_dir = _args.save_dir

    # set up save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = sm.Unet(backbone, encoder_weights='imagenet', input_shape=(
        None, None, 3), classes=2, encoder_freeze=True)
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    # set data generators
    path_train = _args.train_data
    path_train_a = _args.train_annot
    label_map = _args.label_map
    gen = DataGenerator(path_train, path_train_a, label_map,
                        image_size=image_size, batch_size=batch_size)

    path_val = _args.val_data
    path_val_a = _args.val_annot
    val_gen = DataGenerator(path_val, path_val_a, label_map,
                            image_size=image_size, batch_size=batch_size, augment_data=False)
    # set tensorboard callback
    run_logdir = os.path.join(
        save_dir,
        datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime(
            "%Y%m%d-%H%M%S"))
    tensorboard_cb = keras.callbacks.TensorBoard(
        run_logdir, histogram_freq=1, update_freq='epoch')

    # train model
    history = model.fit_generator(
        gen, validation_data=val_gen, verbose=1, epochs=epochs, callbacks=[tensorboard_cb])
    if image_size != None:
        save_str = os.path.join(
            save_dir, f'training_{backbone}_{image_size[0]}_{image_size[1]}_epochs_{epochs}_{datetime.date.today()}.h5')
    else:
        save_str = os.path.join(
            save_dir, f'training_{backbone}_OrigImageSize_epochs_{epochs}_{datetime.date.today()}.h5')
    print(save_str)
    model.save_weights(save_str)

    # save train history
    with open(os.path.join(save_dir, 'train_history.pickle'), 'bw+') as history_file:
        pickle.dump(history.history, history_file)
    # model evaluation
    eval_gen = DataGenerator(args.test, args.test_annot, label_map,
                             image_size=image_size, batch_size=batch_size, augment_data=False)
    score = model.evaluate_generator(eval_gen)
    print("Evaluation scores: ", score)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Shape segmentation model.")
    parser.add_argument("--train_data", type=str,
                        help="directory in which training images are located.", required=True)
    parser.add_argument("--train_annot", type=str,
                        help="directory in which training mask images are located", required=True)
    parser.add_argument("--val_data", type=str,
                        help="directory in which validation images are located.", required=True)
    parser.add_argument("--val_annot", type=str,
                        help="directory in which validation mask images are located", required=True)
    parser.add_argument("--test", type=str,
                        help="directory in which test images are located", required=True)
    parser.add_argument("--test_annot", type=str,
                        help="directory in which test mask images are located", required=True)
    parser.add_argument("--label_map", type=str,
                        help='directory in which label map (exported by CVAT) is located', required=True)
    parser.add_argument("--input_shape", nargs='+',
                        help="Specifiy size of input images h,w", required=True)
    parser.add_argument("--batch_size", type=int,
                        default=20, help="Batch size")
    parser.add_argument("--save_dir", type=str, default="save_dir",
                        help="Directory to save weights of trained model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--backbone", type=str, default='resnet34',
                        help="U-Net backbone")
    parser.add_argument("--model", type=str, default="",
                        help="directory in which trained model is saved")

    args = parser.parse_args()
    train_model(args)
