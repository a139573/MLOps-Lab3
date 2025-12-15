import os
import json
import torch
import mlflow
from mlflow.tracking import MlflowClient

# --- CONFIGURATION ---
MODEL_REGISTRY_NAME = "OxfordPetsMobileNet" # Must match the name in train.py
EXPORT_DIR = "production_models"
ONNX_MODEL_NAME = "model.onnx"

def export_best_model():
    client = MlflowClient()

    # 1. Search for all versions of this registered model
    print(f"Searching for registered versions of '{MODEL_REGISTRY_NAME}'...")
    
    # Filter by name as requested in instructions
    versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")

    if not versions:
        print(f"Error: No registered models found with name '{MODEL_REGISTRY_NAME}'.")
        print("Did you add 'registered_model_name' to your train.py?")
        return

    print(f"Found {len(versions)} versions. Comparing metrics...")

    best_run_id = None
    best_acc = -1.0
    best_version_num = None

    # 2. Iterate through versions to find the one with the highest accuracy
    # Note: ModelVersion objects don't hold metrics, so we must fetch the Run for each version.
    for version in versions:
        run_id = version.run_id
        run = client.get_run(run_id)
        
        # Get accuracy (default to 0 if not found)
        acc = run.data.metrics.get("val_acc", 0.0)
        
        print(f" - Version {version.version} (Run {run_id[:8]}): Accuracy = {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_run_id = run_id
            best_version_num = version.version

    print(f"\n🏆 Best Model: Version {best_version_num} with Accuracy: {best_acc:.4f}")
    print(f"✅ Best Run ID: {best_run_id}")

    # 3. Create Output Directory
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # 4. Download Class Labels (Artifact from the best run)º
    print("Downloading class labels...")
    try:
        local_path = client.download_artifacts(best_run_id, "class_labels.json", dst_path=EXPORT_DIR)
        with open(local_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} class labels.")
    except OSError as e:
        print(f"Warning: Could not download labels. {e}")

    # 5. Load the Best Model (using the run URI)
    print(f"Loading PyTorch model from run {best_run_id}...")
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    model.to("cpu") # Render doesn't support CUDA 
    model.eval()

    # 6. Serialize to ONNX
    print("Exporting model to ONNX format (Opset 18)...")
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(EXPORT_DIR, ONNX_MODEL_NAME)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"🎉 Success! Model saved to: {onnx_path}")

if __name__ == "__main__":
    export_best_model()