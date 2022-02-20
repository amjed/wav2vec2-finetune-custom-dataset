from tensorflow/tensorflow:latest-gpu
RUN pip install --upgrade pip
# NVIDIA GeForce RTX 3060 with CUDA capability sm_86 is not compatible with the current PyTorch installation. The current PyTorch
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install librosa==0.8.1 transformers==4.16.2 numpy==1.22 datasets==1.16.1 jiwer==2.3.0
RUN apt-get -y install awscli

COPY vastai/onstartup.sh /root/onstartup.sh
COPY vocab.json /root/vocab.json
COPY .wheels/* .

RUN pip install --upgrade --no-deps --force-reinstall *.whl
