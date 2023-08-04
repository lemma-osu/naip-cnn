FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY . /app

RUN pip install -e .