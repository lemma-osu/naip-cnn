# Contributing

## Setup

### Using Docker

Docker is the recommended solution for running this project, as GPU installations of Tensorflow can be tricky to setup, especially on Windows. The steps below are for using the Docker image through [VS Code](https://code.visualstudio.com/), but the image can also be run directly [from the command line](https://docs.docker.com/get-started/run-your-own-container/).

1. Install the [Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/containers) in VS Code.
2. If Docker Desktop is not already installed on your machine, run `Dev Containers: Install Docker` from the [command palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette).
3. For Windows users, install and enable WSL 2 for Docker by following [this guide](https://docs.docker.com/desktop/wsl/). The basic steps are outlined below.
    1. Install [WSL 2 from the Microsoft Store](https://apps.microsoft.com/store/detail/windows-subsystem-for-linux/9P9TQF7MRM4R).
    2. Run `wsl --install` to install Ubuntu.
    3. Open Docker Desktop settings and enable Ubuntu via `Resources > WSL Integration`.
    4. Enter a WSL shell using `wsl` and run `docker --version` to confirm that Docker is available.
4. Open this project in VS Code.
5. Run `Dev Containers: Reopen in Container` from the command palette. This will build the Docker image and start a containerized VS Code instance.

### Without Docker

If you do not want to use Docker, you can follow [this guide](https://www.tensorflow.org/install/pip) and then use `pip install -e .` to install the required packages.

### Checking Your Setup

Once your environment is setup, you can check that Tensorflow is working and GPU support is enabled by running by running the code below:

```python
import tensorflow as tf

assert tf.config.list_physical_devices('GPU')
```