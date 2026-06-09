from __future__ import annotations

import os
DEFAULT_DATA_DIR = "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130"
DEFAULT_MODEL_PATH = "/home/renku/work/model-artifacts/mnist-models/mnist_cnn.pt"
DEFAULT_DASHBOARD = "/home/renku/work/pi-gpt-renku-project-2/dashboard/app.py"


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
    os.environ.setdefault("MNIST_DATA_DIR", DEFAULT_DATA_DIR)
    os.environ.setdefault("MNIST_MODEL_PATH", DEFAULT_MODEL_PATH)
    start_marimo()


if __name__ == "__main__":
    main()
