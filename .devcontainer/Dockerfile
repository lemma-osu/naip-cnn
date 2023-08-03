FROM tensorflow/tensorflow:latest-gpu-jupyter

ENV python_version=3.10

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY . /app

RUN pip install -e .