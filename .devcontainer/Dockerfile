FROM python:3.11-slim


# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# Install uv. The version is pinned for reproducibility, not because of any known issues.
# See https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

# Set up the workspace
WORKDIR /workspaces/naip-cnn
COPY . /workspaces/naip-cnn

# Install dependencies with uv such that they're available on the path
ENV UV_PROJECT_ENVIRONMENT=/usr/local
RUN uv sync --locked && \
    pip freeze

# Set up global IPython configuration for the devcontainer
RUN ipython profile create default
RUN rm $HOME/.ipython/profile_default/ipython_config.py
RUN cp ./.devcontainer/ipython_config.json $HOME/.ipython/profile_default/ipython_config.json