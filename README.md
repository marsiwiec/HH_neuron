# Hodgkin-Huxley Neuron with Ornstein-Uhlenbeck Synaptic Input

This project implements a Hodgkin-Huxley neuron model with synaptic input modeled as an Ornstein-Uhlenbeck (OU) process. It should be run as a [marimo](https://marimo.io/) notebook.

## Prerequisites

- **Python 3.13+**
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

## Getting Started

### Using Nix/devenv

If you have `devenv` installed, you can set up the environment automatically:

```bash
# Enter the development environment
devenv shell

# If you have envrc, the environment should automatically load
# every time you enter the project directory after you run direnv allow

# Run the notebook
marimo edit neuron.py
```

### Using other environments

If you don't use Nix, you can use `uv` or `pip` to manage dependencies.

#### Using uv (recommended)

```bash
# Install dependencies and run the notebook
uv run marimo edit neuron.py
```

#### Using pip

```bash
# Install dependencies
pip install .

# Run the notebook
marimo edit neuron.py
```

## Usage

The `neuron.py` file is a marimo notebook. You can interact with it in two ways:

1.  **Edit Mode (Interactive)**:
    ```bash
    marimo edit neuron.py
    ```
    This opens the notebook in your browser, allowing you to modify parameters and see real-time updates.

2.  **Run Mode (App)**:
    ```bash
    marimo run neuron.py
    ```
    This runs the notebook as a read-only web application.

3.  **Script Mode**:
    ```bash
    python neuron.py
    ```
    Running it directly with Python will execute the notebook as a script.
