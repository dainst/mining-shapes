from tensorflow import keras

modelpath = '/home/Data/Models/seg_model.h5'
unet_model = keras.models.load_model(modelpath, compile=False)

unet_model.save(
    '/home/Code/Shape_Similarity/ModelNpmPackage/UNetSegmentation/segmodel.h5')
