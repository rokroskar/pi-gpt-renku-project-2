from __future__ import annotations

import os
import sys
from pathlib import Path

from .train import parse_args, train

DEFAULT_DATA_DIR = "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130"
DEFAULT_MODEL_DIR = "/home/renku/work/model-artifacts/mnist-models"
DEFAULT_OUTPUT_DIR = "/home/renku/work/model-artifacts/mnist-models"
DEFAULT_DASHBOARD = "/home/renku/work/pi-gpt-renku-project-2/dashboard/app.py"


def ensure_model() -> None:
    data_dir = os.environ.get("MNIST_DATA_DIR", DEFAULT_DATA_DIR)
    model_dir = Path(os.environ.get("MNIST_MODEL_DIR", DEFAULT_MODEL_DIR))
    model_path = Path(os.environ.get("MNIST_MODEL_PATH", str(model_dir / "mnist_cnn.pt")))
    output_dir = os.environ.get("MNIST_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

    os.environ.setdefault("MNIST_DATA_DIR", data_dir)
    os.environ.setdefault("MNIST_MODEL_PATH", str(model_path))

    if model_path.exists():
        print(f"Found existing model: {model_path}", flush=True)
        return

    print(f"Model not found at {model_path}; training one from mounted Zenodo data.", flush=True)
    sys.argv = [
        "train",
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--model-dir",
        str(model_path.parent),
        "--target-accuracy",
        os.environ.get("MNIST_TARGET_ACCURACY", "0.99"),
        "--max-epochs",
        os.environ.get("MNIST_MAX_EPOCHS", "20"),
        "--batch-size",
        os.environ.get("MNIST_BATCH_SIZE", "128"),
    ]
    train(parse_args())


def start_marimo() -> None:
    dashboard = os.environ.get("MNIST_DASHBOARD_APP", DEFAULT_DASHBOARD)
    port = os.environ.get("PORT", "8080")
    argv = [
        "marimo",
        "run",
        dashboard,
        "--host",
        "0.0.0.0",
        "--port",
        port,
        "--headless",
        "--no-token",
        "--session-ttl",
        "1800",
    ]
    base_url_path = os.environ.get("RENKU_BASE_URL_PATH", "")
    if base_url_path:
        if not base_url_path.startswith("/"):
            base_url_path = "/" + base_url_path
        argv.extend(["--base-url", base_url_path])
    print("Starting Marimo: " + " ".join(argv), flush=True)
    os.execvp(argv[0], argv)


def main() -> None:
    ensure_model()
    start_marimo()


if __name__ == "__main__":
    main()
