import os
import json
import torch
import mlflow
from mlflow.tracking import MlflowClient

# --- CONFIGURATION ---
EXPERIMENT_NAME = "Oxford_Pets_Transfer_Learning"
EXPORT_DIR = "production_models"  # Where we save the files for the app
ONNX_MODEL_NAME = "model.onnx"
JSON_LABELS_NAME = "class_labels.json"

def export_best_model():
    # 1. Setup Client
    client = MlflowClient()
    
    # 2. Get Experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Error: Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")
        return

    # 3. Search Runs to find the Best Model
    # We sort by 'metrics.val_acc' in Descending order (DESC) to get the highest accuracy first
    print(f"Searching for the best run in experiment '{EXPERIMENT_NAME}'...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_acc DESC"],
        max_results=1
    )

    if not runs:
        print("Error: No runs found.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_acc = best_run.data.metrics.get("val_acc", 0.0)
    
    print(f"✅ Best Run ID: {best_run_id}")
    print(f"🏆 Validation Accuracy: {best_acc:.4f}")

    # 4. Create Output Directory
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # 5. Download Class Labels (Artifact)
    # The training script saved 'class_labels.json' into the run artifacts. We retrieve it now.
    print("Downloading class labels...")
    local_path = client.download_artifacts(best_run_id, "class_labels.json", dst_path=EXPORT_DIR)
    
    # Verify we can read it
    with open(local_path, 'r') as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} class labels.")

    # 6. Load the Best Model
    print("Loading PyTorch model...")
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Prepare for ONNX Export
    # Render is CPU-only, so we ensure the model is on CPU
    model.to("cpu")
    model.eval() # Set to evaluation mode (disable dropout, etc.)

    # 7. Serialize to ONNX
    print(f"Exporting model to ONNX format (Opset 18)...")
    
    # Create a dummy input that matches the input shape (Batch=1, Channels=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    onnx_path = os.path.join(EXPORT_DIR, ONNX_MODEL_NAME)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,  # Standard version
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'}, # Allow variable batch sizes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"🎉 Success! Model saved to: {onnx_path}")
    print(f"📂 Class labels saved to: {local_path}")

if __name__ == "__main__":
    export_best_model()