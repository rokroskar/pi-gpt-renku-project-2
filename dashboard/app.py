from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mnist_job.data import find_mnist_root, load_numpy  # noqa: E402
from mnist_job.model import SmallCNN  # noqa: E402
from mnist_job.train import train as train_model  # noqa: E402

st.set_page_config(page_title="MNIST Renku Demo", page_icon="🔢", layout="wide")

DATA_DIR = Path(os.environ.get(
    "MNIST_DATA_DIR",
    "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130",
))
MODEL_PATH = Path(os.environ.get(
    "MNIST_MODEL_PATH",
    "/home/renku/work/model-artifacts/mnist-models/mnist_cnn.pt",
))
PRETRAINED_MODEL_PATH = Path(os.environ.get(
    "MNIST_PRETRAINED_MODEL_PATH",
    "/home/renku/work/pretrained-model-artifacts/mnist-models/mnist_cnn.pt",
))

st.title("🔢 MNIST non-interactive ML training demo")
st.caption("Data is read from the Renku Zenodo connector; no ad-hoc downloads are used.")

@st.cache_resource
def load_model(path: str):
    model = SmallCNN()
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("metrics", {})


def predict(model, images: np.ndarray):
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    x = (x - 0.1307) / 0.3081
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy()
    return probs

try:
    mnist_root = find_mnist_root(DATA_DIR)
    images, labels = load_numpy(mnist_root, "test")
    st.success(f"Using MNIST data from `{mnist_root}`")
except Exception as exc:  # pragma: no cover - UI path
    st.error(str(exc))
    st.stop()

active_model_path = MODEL_PATH
if not active_model_path.exists() and PRETRAINED_MODEL_PATH.exists():
    active_model_path = PRETRAINED_MODEL_PATH
    st.info(f"Using pretrained model from read-only connector: `{active_model_path}`")

if not active_model_path.exists():
    st.warning(
        f"No model found at writable connector path `{MODEL_PATH}` or "
        f"pretrained connector path `{PRETRAINED_MODEL_PATH}`."
    )
    if st.button("Train model now", type="primary"):
        with st.spinner("Training until test accuracy reaches 0.99..."):
            import argparse
            train_model(argparse.Namespace(
                data_dir=str(mnist_root), output_dir=str(MODEL_PATH.parent), model_dir=str(MODEL_PATH.parent),
                target_accuracy=0.99, max_epochs=20, batch_size=128, lr=3e-3,
                weight_decay=1e-4, seed=42, num_workers=2, cpu=True,
            ))
        st.cache_resource.clear()
        st.rerun()
    st.stop()

model, metrics = load_model(str(active_model_path))
cols = st.columns(4)
cols[0].metric("Best test accuracy", f"{metrics.get('test_accuracy', float('nan')):.4f}")
cols[1].metric("Epoch", metrics.get("epoch", "n/a"))
cols[2].metric("Test loss", f"{metrics.get('test_loss', float('nan')):.4f}")
cols[3].metric("Model", str(active_model_path))

sample_count = st.slider("Number of samples", 4, 24, 12)
seed = st.number_input("Random seed", min_value=0, value=7, step=1)
rng = random.Random(seed)
indices = rng.sample(range(len(images)), sample_count)
sample_images = images[indices]
sample_labels = labels[indices]
probs = predict(model, sample_images)
preds = probs.argmax(axis=1)

st.subheader("Predictions")
for row_start in range(0, sample_count, 4):
    row_cols = st.columns(4)
    for col, idx in zip(row_cols, range(row_start, min(row_start + 4, sample_count))):
        with col:
            st.image(sample_images[idx], width=120, clamp=True)
            ok = "✅" if preds[idx] == sample_labels[idx] else "❌"
            st.markdown(f"### {ok} predicted `{preds[idx]}` / true `{sample_labels[idx]}`")
            st.bar_chart(pd.DataFrame({"probability": probs[idx]}, index=list(range(10))))
