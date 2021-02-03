from tensorflow import keras
import tensorflow as tf

# resnet34
modelpath = '/home/Data/Models/seg_model.h5'
unet_model = keras.models.load_model(modelpath, compile=False)

layer = keras.layers.Lambda(lambda x: tf.math.argmax(
    x, axis=3)*255, name='foreground_extractor')(unet_model.output)

model = keras.Model(inputs=[unet_model.input],
                    outputs=layer, name='ceramics_segmodel')

model.save(
    '/home/Code/Shape_Similarity/ModelNpmPackage/UNetSegmentation/segmodel.h5')
