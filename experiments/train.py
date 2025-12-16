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
DEFAULT_CONFIG = {
    "batch_size": 32,
    "epochs": 5,
    "lr": 0.001,
    "seed": 123,
    "architecture": "mobilenet_v2" # Default
}

EXPERIMENT_NAME = "Oxford_Pets_Transfer_Learning"
# Generic name so we can compare different architectures in the same registry
MODEL_REGISTRY_NAME = "OxfordPetsModel" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. DATA PREPARATION ---
def prepare_data(batch_size, seed):
    # Standard ImageNet transforms work for ResNet, MobileNet, and ShuffleNet
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    full_dataset = datasets.OxfordIIITPet(root='./data', split='trainval', target_types='category', download=True, transform=transform)
    class_labels = full_dataset.classes
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, class_labels

# --- 3. MODEL SETUP (UPDATED) ---
def build_model(num_classes, architecture):
    print(f"Building model architecture: {architecture}...")
    
    if architecture == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
        # Freeze
        for param in model.parameters():
            param.requires_grad = False
        # Replace Head (MobileNet uses 'classifier')
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        # Freeze
        for param in model.parameters():
            param.requires_grad = False
        # Replace Head (ResNet uses 'fc')
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif architecture == "shufflenet_v2":
        weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        model = models.shufflenet_v2_x1_0(weights=weights)
        # Freeze
        for param in model.parameters():
            param.requires_grad = False
        # Replace Head (ShuffleNet uses 'fc')
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model.to(device)

# --- 4. TRAINING LOOP ---
def train_model(architecture, batch_size, learning_rate, epochs, seed=123):
    set_seed(seed)
    
    # Include architecture in run name for easy ID in MLflow UI
    run_name = f"{architecture}_BS{batch_size}_LR{learning_rate}"
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_architecture": architecture,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed
        })

        train_loader, val_loader, class_labels = prepare_data(batch_size, seed)
        
        labels_path = "class_labels.json"
        with open(labels_path, "w", encoding='utf-8') as f:
            json.dump(class_labels, f)
        mlflow.log_artifact(labels_path)

        # Pass architecture to builder
        model = build_model(len(class_labels), architecture)
        
        criterion = nn.CrossEntropyLoss()
        
        # Handle optimizer params based on architecture head name
        if architecture == "mobilenet_v2":
            params_to_optimize = model.classifier.parameters()
        else:
            params_to_optimize = model.fc.parameters()
            
        optimizer = optim.Adam(params_to_optimize, lr=learning_rate)

        train_loss_history = []
        val_loss_history = []

        print("Starting training...")
        for epoch in range(epochs):
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

            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc
            }, step=epoch)

        # Plot
        fig = plt.figure()
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Val Loss')
        plt.legend()
        plt.title('Loss Curve')
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

        print(f"Registering model under name: {MODEL_REGISTRY_NAME}")
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            name="model",
            registered_model_name=MODEL_REGISTRY_NAME
        )
        print("Training Complete.")

if __name__ == "__main__":
    train_model(
        batch_size=DEFAULT_CONFIG["batch_size"],
        learning_rate=DEFAULT_CONFIG["lr"],
        epochs=DEFAULT_CONFIG["epochs"],
        architecture=DEFAULT_CONFIG["architecture"],
        seed=DEFAULT_CONFIG["seed"]
    )