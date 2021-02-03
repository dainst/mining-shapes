from tensorflow import keras
import tensorflowjs as tfjs


model_path = '/Users/mkihm/Documents/mining-shapes/Shape_Similarity/ModelNpmPackage/Resnet50'
input_shape = (512, 512, 3)

model = keras.applications.resnet50.ResNet50(
    input_shape=input_shape, include_top=False, pooling='avg', weights='imagenet')
model.save('/home/Code/Shape_Similarity/ModelNpmPackage/Resnet50/resnet50.h5')
