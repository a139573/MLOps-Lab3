import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & REPRODUCIBILITY ---
SEED = 123
BATCH_SIZE = 32
EPOCHS = 5  # Small number of epochs as per instructions
LEARNING_RATE = 0.001
EXPERIMENT_NAME = "Oxford_Pets_Transfer_Learning"
RUN_NAME = f"MobileNetV2_BS{BATCH_SIZE}_LR{LEARNING_RATE}"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. DATA PREPARATION ---
def prepare_data():
    print("Downloading and preparing data...")
    
    # Define transforms (Resize to 224x224 as required)
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet stats
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Download Dataset (We use 'trainval' split from OxfordIIITPet)
    # This downloads to a local 'data' folder
    full_dataset = datasets.OxfordIIITPet(root='./data', split='trainval', target_types='category', download=True, transform=transform)
    
    # Get Class Labels (Index to Name)
    class_labels = full_dataset.classes
    
    # Split 80/20 Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, class_labels

# --- 3. MODEL SETUP (TRANSFER LEARNING) ---
def build_model(num_classes):
    print("Building MobileNetV2 model...")
    # Load pre-trained weights
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    # Freeze parameters (Feature Extractor)
    for param in model.parameters():
        param.requires_grad = False

    # Modify Classifier (Final Layer)
    # MobileNetV2 classifier is a Sequential block, last layer is index 1
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model.to(device)

# --- 4. TRAINING LOOP WITH MLFLOW ---
def train_model():
    # 4.1 Setup MLflow Experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=RUN_NAME):
        # Log Parameters
        mlflow.log_params({
            "model": "MobileNetV2",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "dataset": "OxfordIIITPet"
        })

        # Load Data
        train_loader, val_loader, class_labels = prepare_data()
        
        # Save Class Labels as JSON artifact
        labels_path = "class_labels.json"
        with open(labels_path, "w") as f:
            json.dump(class_labels, f)
        mlflow.log_artifact(labels_path)

        # Build Model
        model = build_model(len(class_labels))
        
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

        # Lists for plotting
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        print("Starting training...")
        for epoch in range(EPOCHS):
            # --- Training Phase ---
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct / total
            train_loss_history.append(epoch_train_loss)
            train_acc_history.append(epoch_train_acc)

            # --- Validation Phase ---
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = correct_val / total_val
            val_loss_history.append(epoch_val_loss)
            val_acc_history.append(epoch_val_acc)

            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

            # Log Metrics per Epoch
            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc
            }, step=epoch)

        # --- 5. LOGGING ARTIFACTS ---
        # Plot Loss Curve
        fig = plt.figure()
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Val Loss')
        plt.legend()
        plt.title('Loss Curve')

        # ✅ NEW WAY: Upload directly from memory
        mlflow.log_figure(fig, "loss_curve.png")
        
        # Close the plot to free memory
        plt.close(fig)

        # Log Model (Registering it)
        print("Logging model to MLflow...")
        mlflow.pytorch.log_model(pytorch_model=model, name="model")
        print("Training Complete. Run 'mlflow ui' to view results.")

if __name__ == "__main__":
    train_model()