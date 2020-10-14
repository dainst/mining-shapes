# install build utilities
apt-get update && apt-get install -y gcc make apt-transport-https ca-certificates build-essential

apt-get install -y git
apt-get update && apt-get install -y libgl1-mesa-dev
apt-get install -y graphviz
apt-get install -y tesseract-ocr
apt-get install -y curl unzip
apt-get update && apt-get install -y poppler-utils

# Install object detection api dependencies
DEBIAN_FRONTEND=noninteractive apt-get install -y protobuf-compiler python3-pil python3-lxml python3-tk