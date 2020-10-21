#!/bin/bash

git clone https://github.com/tensorflow/models.git

# Install pycocoapi
git clone --depth 1 https://github.com/cocodataset/cocoapi.git 
cd cocoapi/PythonAPI 
make -j8
cp -r pycocotools /models/research 
cd ../../
rm -rf cocoapi

# Get protoc 3.0.0, rather than the old version already in the container
curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" 
unzip protoc-3.0.0-linux-x86_64.zip -d proto3
mv proto3/bin/* /usr/local/bin
mv proto3/include/* /usr/local/include
rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

# Run protoc on the object detection repo
cd models/research
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py .
python -m pip install --use-feature=2020-resolver .

# Set the PYTHONPATH to finish installing the API
PYTHONPATH=$PYTHONPATH:/models/research/slim
PYTHONPATH=$PYTHONPATH:/models/research

#Run object detection API unit tests
python object_detection/builders/model_builder_tf1_test.py
