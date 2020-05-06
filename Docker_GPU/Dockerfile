FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y git

RUN apt-get install -y tesseract-ocr

RUN apt-get install -y python3 && apt-get install -y python3-pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

# Installing the Tensorflow Object Detection API
RUN git clone https://github.com/tensorflow/models.git

# Install object detection api dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y protobuf-compiler python3-pil python3-lxml python3-tk && \
    pip3 install Cython && \
    pip3 install contextlib2 && \
    pip3 install jupyter && \
    pip3 install matplotlib && \
    pip3 install Pillow && \
    pip3 install tensorflow-gpu==1.15.0 && \
    pip3 install opencv-python && \
    pip3 install pytesseract && \
    pip3 install pandas && \
    pip3 install pylint

# Install pycocoapi
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make -j8 && \
    cp -r pycocotools /models/research && \
    cd ../../ && \
    rm -rf cocoapi

# Get protoc 3.0.0, rather than the old version already in the container
RUN apt-get install -y curl unzip

RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

# Run protoc on the object detection repo
RUN cd models/research && \
    protoc object_detection/protos/*.proto --python_out=.

# Set the PYTHONPATH to finish installing the API
ENV PYTHONPATH=$PYTHONPATH:/models/research/slim
ENV PYTHONPATH=$PYTHONPATH:/models/research

#Run object detection API unit tests
RUN python /models/research/object_detection/builders/model_builder_test.py

#install segmentation models (see https://github.com/qubvel/segmentation_models)
RUN pip install -U segmentation-models==0.2.1

# Entrypoint
WORKDIR /models/research/object_detection
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
