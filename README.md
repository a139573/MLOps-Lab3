# MLOps Lab 3: Model Versioning & Experiment Tracking

> **End-to-End Deep Learning Pipeline with MLflow, Docker, and Transfer Learning.**

This repository transitions the previous project from a random predictor to a robust Deep Learning pipeline capable of classifying pet breeds using **MobileNetV2** and **ShuffleNetV2**.

---

## 🚀 Quick Start

### 1. Installation
This project uses `uv` for dependency management.
```bash
uv sync
```
### 2. Training (Reproduce Experiments)
Run the full hyperparameter search (MobileNet vs. ShuffleNet):

```bash
uv run experiments/run_experiments.py
```

This will train multiple models, log metrics to MLflow, and automatically export the best one.

### 3. View Results (MLflow UI)
To see the leaderboards and loss curves:

```bash
uv run mlflow ui
```
### 4. Run the API (Inference)
Once a model is exported to production_models/model.onnx:


```bash
uv run api/api.py
```
