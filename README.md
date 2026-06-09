# MNIST non-interactive ML jobs on Renku

A compact Renku project that demonstrates how to run reproducible, non-interactive machine-learning jobs from a mounted data connector and serve the resulting model through a Marimo dashboard.

The demo trains a small convolutional neural network on MNIST. The data is provided by a Renku Zenodo connector for DOI [`10.5281/zenodo.10058130`](https://doi.org/10.5281/zenodo.10058130); the code intentionally does **not** download MNIST at runtime.

## What this project shows

- **Connector-backed data access**: training reads canonical MNIST gzip files from the Renku-mounted Zenodo connector.
- **Non-interactive training**: the training launcher runs to completion without a notebook or terminal session.
- **Early stopping by target metric**: training terminates once test accuracy reaches `>= 0.99`.
- **Persistent model artifacts**: the trained model is written to the writable Renku artifact connector.
- **Reusable image builds**: dependencies are installed from `requirements.txt` and reused by Renku launchers.
- **Interactive model inspection**: a Marimo dashboard visualizes predictions from an existing model artifact and offers a manual retraining button.

## Renku project launchers

This project is intentionally kept to three launchers:

| Launcher | Purpose |
| --- | --- |
| **Build reusable Python image (ttyd, marimo)** | Builds the reusable runtime image from this repository's requirements. |
| **Train MNIST CNN direct Python** | Runs the non-interactive training job and writes model artifacts. |
| **MNIST Marimo dashboard** | Starts the Marimo dashboard using the existing model artifact. |

## Data and artifact locations on Renku

The Renku launchers are configured for these connector mount paths:

| Path | Meaning |
| --- | --- |
| `/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130` | Read-only MNIST data connector from Zenodo. |
| `/home/renku/work/model-artifacts/mnist-models` | Writable model artifact directory. |
| `/home/renku/work/model-artifacts/mnist-models/mnist_cnn.pt` | Saved PyTorch checkpoint used by the dashboard. |

The data loader searches the configured data directory for the four canonical MNIST files:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

If they are not present, the job fails with a clear error instead of downloading data.

## Running locally

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train a model from a local copy of the Zenodo MNIST files:

```bash
python -m src.mnist_job.train \
  --data-dir /path/to/mnist-data \
  --model-dir models \
  --output-dir outputs \
  --target-accuracy 0.99 \
  --max-epochs 20
```

Start the Marimo dashboard:

```bash
MNIST_DATA_DIR=/path/to/mnist-data \
MNIST_MODEL_PATH=models/mnist_cnn.pt \
marimo run dashboard/app.py \
  --host 0.0.0.0 \
  --port 8080 \
  --headless \
  --no-token
```

Open the smaller Marimo notebook example:

```bash
marimo edit notebooks/marimo_mnist_example.py
```

## Training outputs

The training script writes:

- `mnist_cnn.pt` — PyTorch checkpoint containing model weights, metrics, data path, and normalization metadata.
- `metrics.json` — full per-epoch training, validation, and test metrics.
- JSON log lines — convenient for Renku job logs and downstream parsing.

A successful run prints a final line like:

```text
FINAL_METRICS {"epoch": 6, "test_accuracy": 0.9917, ...}
```

## Project structure

```text
.
├── dashboard/
│   └── app.py                      # Marimo inference dashboard
├── notebooks/
│   └── marimo_mnist_example.py     # Small Marimo notebook example
├── src/mnist_job/
│   ├── data.py                     # MNIST connector data loading
│   ├── model.py                    # Small CNN architecture
│   ├── train.py                    # Non-interactive training entry point
│   └── dashboard_launcher.py       # Renku dashboard wrapper for proxy-aware Marimo startup
├── requirements.txt                # Runtime dependencies for Renku image builds
├── Procfile                        # Simple web entry point
└── README.md
```

## Configuration

Useful environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MNIST_DATA_DIR` | `/data/mnist` for direct training; Renku connector path in launcher wrapper | Directory containing MNIST gzip files. |
| `MNIST_MODEL_DIR` | `models` or Renku artifact path | Directory for saved model checkpoints. |
| `MNIST_MODEL_PATH` | `<model-dir>/mnist_cnn.pt` | Checkpoint loaded by the dashboard. |
| `MNIST_TARGET_ACCURACY` | `0.99` | Target accuracy used by the training script and manual dashboard retraining. |
| `MNIST_MAX_EPOCHS` | `20` | Maximum epochs before failing if target accuracy is not reached. |
| `PORT` | `8080` | Dashboard port. |
| `RENKU_BASE_URL_PATH` | unset | Used by the Renku launcher so Marimo works behind the session proxy. |

## Notes for Renku users

- Use the image-build launcher after changing dependencies in `requirements.txt`.
- Use the training launcher to create or refresh `mnist_cnn.pt` in the writable artifact connector.
- Use the Marimo dashboard launcher to inspect predictions interactively.
- The dashboard launcher starts Marimo behind the Renku session proxy and does not train automatically on startup.
- The training job exits successfully only when the target accuracy is reached; otherwise it fails explicitly.
