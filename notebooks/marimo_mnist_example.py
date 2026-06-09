import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import os
    import random
    import sys

    import marimo as mo
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from mnist_job.data import find_mnist_root, load_numpy
    from mnist_job.model import SmallCNN

    DATA_DIR = Path(
        os.environ.get(
            "MNIST_DATA_DIR",
            "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130",
        )
    )
    MODEL_PATH = Path(
        os.environ.get(
            "MNIST_MODEL_PATH",
            "/home/renku/work/model-artifacts/mnist-models/mnist_cnn.pt",
        )
    )
    return DATA_DIR, MODEL_PATH, SmallCNN, find_mnist_root, load_numpy, mo, np, random, torch


@app.cell
def _(mo):
    mo.md(
        r"""
# MNIST connector notebook example

This is a small Marimo notebook example for the Renku MNIST demo. It reads MNIST
from the mounted Zenodo data connector and, if a trained model artifact is
available, shows predictions for random test images.

It does **not** train automatically. Run the training launcher first if the model
artifact is missing.
        """
    )
    return


@app.cell
def _(DATA_DIR, find_mnist_root, load_numpy, mo):
    try:
        mnist_root = find_mnist_root(DATA_DIR)
        images, labels = load_numpy(mnist_root, "test")
        data_status = mo.md(f"✅ MNIST data found at `{mnist_root}`")
    except Exception as exc:
        mnist_root = None
        images, labels = None, None
        data_status = mo.md(f"❌ MNIST data not available: `{exc}`")
    data_status
    return images, labels, mnist_root


@app.cell
def _(MODEL_PATH, SmallCNN, mo, torch):
    model = None
    metrics = None
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model = SmallCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        metrics = checkpoint.get("metrics", {})
        model_status = mo.md(f"✅ Loaded model artifact from `{MODEL_PATH}`")
    else:
        model_status = mo.md(f"⚠️ Model artifact not found at `{MODEL_PATH}`")
    model_status
    return metrics, model


@app.cell
def _(metrics, mo):
    if metrics:
        mo.hstack(
            [
                mo.stat(label="Epoch", value=str(metrics.get("epoch", "n/a"))),
                mo.stat(
                    label="Test accuracy",
                    value=f"{metrics.get('test_accuracy', 0):.4f}",
                ),
            ]
        )
    return


@app.cell
def _(mo):
    count = mo.ui.slider(4, 16, value=8, label="Number of random test images")
    seed = mo.ui.number(value=7, label="Random seed")
    mo.vstack([count, seed])
    return count, seed


@app.cell
def _(count, images, labels, model, mo, np, random, seed, torch):
    if images is None or labels is None:
        output = mo.md("No MNIST data available.")
    elif model is None:
        output = mo.md("No model artifact available. Run the training launcher first.")
    else:
        rng = random.Random(int(seed.value))
        idxs = rng.sample(range(len(images)), int(count.value))
        x = torch.tensor(images[idxs], dtype=torch.float32).unsqueeze(1) / 255.0
        x = (x - 0.1307) / 0.3081
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).numpy()
        preds = probs.argmax(axis=1)
        rows = []
        for i, idx in enumerate(idxs):
            rows.append(
                mo.vstack(
                    [
                        mo.image(np.repeat(images[idx][:, :, None], 3, axis=2), width=96),
                        mo.md(
                            f"prediction: **{int(preds[i])}**  "
                            f"true: **{int(labels[idx])}**"
                        ),
                    ]
                )
            )
        output = mo.hstack(rows, wrap=True)
    output
    return


if __name__ == "__main__":
    app.run()
