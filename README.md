# Renku MNIST non-interactive training demo

This project demonstrates a reusable Renku workflow for running non-interactive ML training jobs.

- Data comes from a Renku Zenodo data connector for DOI `10.5281/zenodo.10058130`.
- The training script refuses to download MNIST ad-hoc; it must find MNIST files in the mounted connector.
- The non-interactive job trains a small CNN and stops as soon as test accuracy reaches the requested threshold (`0.99` by default).
- The dashboard launcher starts a Marimo app and visualizes model predictions. If no trained model is present, it can retrain one from the mounted Zenodo data.

## Expected Renku launchers

### Train MNIST CNN job

Command:

```bash
python -m mnist_job.train --data-dir /data/mnist --output-dir outputs --target-accuracy 0.99 --max-epochs 20
```

### MNIST inference dashboard

Command:

```bash
marimo run dashboard/app.py --host 0.0.0.0 --port 8080 --headless --no-token
```

Set `MNIST_DATA_DIR` if the Zenodo connector is mounted somewhere other than `/data/mnist`.
