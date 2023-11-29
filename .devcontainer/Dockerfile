FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY . /app

RUN pip install -e .
RUN python ./scripts/setup_env.py