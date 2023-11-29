import json
from pathlib import Path


def set_ipython_config(config_dict):
    """Write a configuration dict to the default profile for ipython.

    See https://ipython.readthedocs.io/en/stable/config/intro.html
    """
    ipython_config_path = Path.home() / ".ipython/profile_default/ipython_config.json"

    if not ipython_config_path.exists():
        ipython_config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ipython_config_path, "w") as f:
        json.dump(config_dict, f, indent=4)


if __name__ == "__main__":
    # Auto-reload modules by default
    config = {
        "InteractiveShellApp": {
            "extensions": [
                "autoreload",
            ],
            "exec_lines": [
                "%autoreload 2",
            ],
        },
    }

    set_ipython_config(config)
