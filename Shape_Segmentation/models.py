from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Softmax, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K




def unet(input_shape, classes):
    '''
    Params: input_shape -- the shape of the images that are input to the model
                           in the form (width_or_height, width_or_height,
                           num_color_channels)
    Returns: model -- a model that has been defined, but not yet compiled.
                      The model is an implementation of the Unet paper
                      (https://arxiv.org/pdf/1505.04597.pdf) and comes
                      from this repo https://github.com/zhixuhao/unet. It has
                      been modified to keep up with API changes in keras 2.
    '''
    inputs = Input(input_shape)

    conv1 = Conv2D(filters=64,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(filters=64,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=128,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(filters=128,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=256,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(filters=256,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters=512,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(filters=512,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters=1024,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(filters=1024,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = Conv2D(filters=512,
                 kernel_size=2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up6)
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(filters=512,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(filters=512,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(filters=256,
                 kernel_size=2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up7)
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(filters=256,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(filters=256,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(filters=128,
                 kernel_size=2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up8)
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(filters=128,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(filters=128,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(filters=64,
                 kernel_size=2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up9)
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(filters=64,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(filters=64,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(filters=2,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(filters=classes, kernel_size=1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model