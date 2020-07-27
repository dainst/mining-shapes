# Normalize Shape

## Point Detector
Neural network to detect certain keypoints on profile drawing. The detection network is based on the archticture developed by [Bin Xiao et al.](https://arxiv.org/pdf/1804.06208.pdf).
### Data Format
To train the network provide rgb images with corresponding keypoints. The keypoints should be dumped in [CVAT xml 1.1 format](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md). 

### Use the code
First, mount data in docker-compose file. Next, use the following code snippet to run training.
```python
from point_detector import PointDetector, DataGeneratorInputs

path = '/home/images/mining_shapes/preliminary/results/'
keypoints = '/home/images/OUTPUT/DumpedPoints/43_ProfilePoint_2.xml'
val_path = '/home/images/OUTPUT/CVAT_Point_val_600'
val_keypoints = '/home/images/OUTPUT/DumpedPoints/41_ProfilePointDetection.xml'
epochs = 60
input_shape = (256, 256, 3)


model = PointDetector(DataGeneratorInputs(path, keypoints),
                        DataGeneratorInputs(val_path, val_keypoints), input_shape= input_shape, batch_size=16,sigma=3)
model.fit(epochs=epochs)
```
Use tensorboard to see training progress. Tensorboard can be started with the following bash command.
```bash
tensorboard --logdir save_dir
```
## Trained model
Trained model can be downloaded from the following [link](https://cumulus.dainst.org/index.php/s/DxpTApRC2HGbAwd).