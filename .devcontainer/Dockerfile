FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY . /app

RUN pip install -e .

RUN ipython profile create default
RUN rm $HOME/.ipython/profile_default/ipython_config.py
RUN cp ./.devcontainer/ipython_config.json $HOME/.ipython/profile_default/ipython_config.json