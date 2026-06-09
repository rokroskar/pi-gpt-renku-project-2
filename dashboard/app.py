import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import argparse
    import os
    import random
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from mnist_job.data import find_mnist_root, load_numpy
    from mnist_job.model import SmallCNN
    from mnist_job.train import train as train_model

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
    PRETRAINED_MODEL_PATH = Path(
        os.environ.get(
            "MNIST_PRETRAINED_MODEL_PATH",
            "/home/renku/work/pretrained-model-artifacts/mnist-models/mnist_cnn.pt",
        )
    )
    return (
        DATA_DIR,
        MODEL_PATH,
        PRETRAINED_MODEL_PATH,
        SmallCNN,
        argparse,
        find_mnist_root,
        load_numpy,
        mo,
        np,
        random,
        torch,
        train_model,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
# 🔢 MNIST non-interactive ML training demo

This Marimo dashboard reads MNIST from the Renku Zenodo connector and
visualizes predictions from a CNN trained by a non-interactive Renku job.
        """
    )
    return


@app.cell
def _(DATA_DIR, find_mnist_root, load_numpy, mo):
    try:
        mnist_root = find_mnist_root(DATA_DIR)
        images, labels = load_numpy(mnist_root, "test")
        data_status = mo.md(f"✅ Using MNIST data from `{mnist_root}`")
    except Exception as exc:
        mnist_root = None
        images, labels = None, None
        data_status = mo.md(f"❌ Could not load MNIST data: `{exc}`")
    data_status
    return data_status, images, labels, mnist_root


@app.cell
def _(MODEL_PATH, PRETRAINED_MODEL_PATH, mo):
    active_model_path = MODEL_PATH
    model_source = "writable model artifact connector"
    if not active_model_path.exists() and PRETRAINED_MODEL_PATH.exists():
        active_model_path = PRETRAINED_MODEL_PATH
        model_source = "read-only pretrained artifact connector"

    if active_model_path.exists():
        model_status = mo.md(f"✅ Using model from `{active_model_path}` ({model_source})")
    else:
        model_status = mo.md(
            f"⚠️ No model found at writable path `{MODEL_PATH}` or pretrained path `{PRETRAINED_MODEL_PATH}`. "
            "Use the retraining button below, or run the training job launcher first."
        )
    model_status
    return active_model_path, model_source, model_status


@app.cell
def _(mo):
    retrain = mo.ui.run_button(label="Retrain model in writable artifact connector")
    controls = mo.md(
        """
        ## Controls

        Samples to show: {sample_count}

        Random seed: {seed}

        {retrain}
        """
    ).batch(
        sample_count=mo.ui.slider(4, 24, step=4, value=12, show_value=True),
        seed=mo.ui.number(value=7, start=0, step=1),
        retrain=retrain,
    )
    controls
    return controls, retrain


@app.cell
def _(
    MODEL_PATH,
    argparse,
    controls,
    mnist_root,
    mo,
    retrain,
    train_model,
):
    retrain_status = None
    if retrain.value:
        if mnist_root is None:
            retrain_status = mo.md("❌ Cannot retrain because MNIST data is not available.")
        else:
            retrain_status = mo.md("⏳ Training model until test accuracy reaches 0.99 ...")
            train_model(
                argparse.Namespace(
                    data_dir=str(mnist_root),
                    output_dir=str(MODEL_PATH.parent),
                    model_dir=str(MODEL_PATH.parent),
                    target_accuracy=0.99,
                    max_epochs=20,
                    batch_size=128,
                    lr=3e-3,
                    weight_decay=1e-4,
                    seed=42,
                    num_workers=2,
                    cpu=True,
                )
            )
            retrain_status = mo.md(f"✅ Training complete; wrote model to `{MODEL_PATH}`.")
    retrain_status if retrain_status is not None else mo.md("")
    return (retrain_status,)


@app.cell
def _(SmallCNN, active_model_path, torch):
    if active_model_path.exists():
        model = SmallCNN()
        checkpoint = torch.load(active_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        metrics = checkpoint.get("metrics", {})
    else:
        model = None
        metrics = {}
    return checkpoint if active_model_path.exists() else None, metrics, model


@app.cell
def _(active_model_path, metrics, mo):
    mo.hstack(
        [
            mo.stat(
                label="Best test accuracy",
                value=f"{metrics.get('test_accuracy', float('nan')):.4f}",
                bordered=True,
            ),
            mo.stat(label="Epoch", value=str(metrics.get("epoch", "n/a")), bordered=True),
            mo.stat(
                label="Test loss",
                value=f"{metrics.get('test_loss', float('nan')):.4f}",
                bordered=True,
            ),
            mo.stat(label="Model path", value=str(active_model_path), bordered=True),
        ],
        widths="equal",
    )
    return


@app.cell
def _(controls, images, labels, model, np, random, torch):
    if model is None or images is None or labels is None:
        selected_images, selected_labels, probs, preds = None, None, None, None
    else:
        sample_count = int(controls.value["sample_count"])
        seed = int(controls.value["seed"])
        rng = random.Random(seed)
        indices = rng.sample(range(len(images)), sample_count)
        selected_images = images[indices]
        selected_labels = labels[indices]
        x = torch.tensor(selected_images, dtype=torch.float32).unsqueeze(1) / 255.0
        x = (x - 0.1307) / 0.3081
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs, selected_images, selected_labels


@app.cell
def _(mo, preds, probs, selected_images, selected_labels):
    def probability_bars(probabilities):
        bars = []
        for digit, probability in enumerate(probabilities):
            width = float(probability) * 100
            bars.append(
                f"""
                <div style='display:flex;align-items:center;gap:0.4rem;margin:0.15rem 0;'>
                  <span style='width:1.2rem;text-align:right;font-variant-numeric:tabular-nums;'>{digit}</span>
                  <div style='flex:1;background:#edf2f7;border-radius:999px;height:0.55rem;overflow:hidden;'>
                    <div style='width:{width:.1f}%;height:100%;background:#2563eb;'></div>
                  </div>
                  <span style='width:3rem;font-size:0.8rem;font-variant-numeric:tabular-nums;'>{probability:.2f}</span>
                </div>
                """
            )
        return "".join(bars)

    if selected_images is None:
        output = mo.md("No predictions to show yet; train or mount a model first.")
    else:
        cards = []
        for image, label, pred, probability in zip(selected_images, selected_labels, preds, probs):
            ok = "✅" if int(pred) == int(label) else "❌"
            cards.append(
                mo.Html(
                    f"""
                    <div style='border:1px solid #d8dee9;border-radius:12px;padding:0.75rem;margin:0.25rem;background:white;'>
                      <div style='font-size:1.05rem;font-weight:700;margin-bottom:0.25rem;'>{ok} predicted {int(pred)} / true {int(label)}</div>
                      {mo.image(image, width=120)}
                      <div style='margin-top:0.5rem;'>{probability_bars(probability)}</div>
                    </div>
                    """
                )
            )
        rows = [
            mo.hstack(cards[start : start + 4], widths="equal")
            for start in range(0, len(cards), 4)
        ]
        output = mo.vstack([mo.md("## Predictions"), *rows])
    output
    return output, probability_bars


if __name__ == "__main__":
    app.run()
