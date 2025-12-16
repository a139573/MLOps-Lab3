# run_pipeline.py
# import itertools
from train import train_model
from export_model import export_best_model

def main():
    architectures = ["shufflenet_v2", "mobilenet_v2", "resnet18"]
    lrs = [0.001, 0.0001]
    batch_sizes = [16, 32]
    n_epochs = [5, 10] 
    
    for arch in architectures:
        for lr in lrs:
            for bs in batch_sizes:
                for epochs in n_epochs:
                    print(f"Training with ARCH={arch}, LR={lr}, BS={bs}, epochs={epochs}")
                    train_model(architecture=arch, batch_size=bs, learning_rate=lr, epochs=epochs)

    # 2. Automatically Pick and Export the Winner
    print("\n🏁 All training runs complete. Exporting best model...")
    export_best_model()

if __name__ == "__main__":
    main()