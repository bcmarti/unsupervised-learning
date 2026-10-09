# unsupervised-learning

## Introduction

This project sets up a Python environment for conducting unsupervised learning experiments using:

* **MOMENT** (a time-series foundation model)
* **3W Toolkit** (utilities for the Petrobras 3W dataset)

The main goal is to fine-tune MOMENT using unlabeled data similar to the **3W Dataset** for signal reconstruction. The underlying assumption is that when the model encounters anomalous data, its reconstruction will deviate from the actual signal, indicating a possible failure.

The model will be validated and tested using the labeled **3W Dataset**.

## Prerequisites

Before getting started, make sure that **uv** is installed by following the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

**uv** will be used to manage the Python virtual environment and dependencies, ensuring a reproducible development environment.

## Environment setup

Open a prompt window in this repository folder and run the following commands according to your operating system:

- ### Linux / macOS (terminal):

```bash
uv venv .venv
source .venv/bin/activate
```

- ### Windows (cmd):

```bash
uv venv .venv
.venv\Scripts\activate.bat
```

- ### Windows (PowerShell):

```bash
uv venv .venv
.venv\Scripts\Activate.ps1
``` 

- ### Windows (Bash):

```bash
uv venv .venv
source .venv/Scripts/activate
```

Now, use this command to setup the virtual environment with the dependencies specified in the [pyproject.toml](pyproject.toml):

```bash
uv sync --all-extras
```

## Verifying the installation

You can quickly test that everything is working:

```bash
python - << 'EOF'
import momentfm
import ThreeWToolkit
from importlib.metadata import version

print("Setup successful")
print("MOMENT version:", version('momentfm'))
print("3W Toolkit version:", ThreeWToolkit.__version__)
EOF
```

## Downloading the 3W Dataset

To download the **3W Dataset 2.0.0**, open a new prompt window in this repository folder and activate the virtual environment. Then run the script:

```bash
python utils/dataset_download.py
```

The dataset will be downloaded in `dataset` folder.

## Troubleshooting

If you encounter `ModuleNotFoundError` or need to install an additional package in the virtual environment, use the following command:

```bash
uv add <package>
```

This will add the package to the project's dependencies and update the environment accordingly.