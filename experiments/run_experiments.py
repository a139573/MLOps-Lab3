# run_pipeline.py
import itertools
from train import train_model
from export_model import export_best_model

def main():
    # 1. Run all Training variations
    lrs = [0.001, 0.0001]
    batch_sizes = [32]
    
    for lr in lrs:
        for bs in batch_sizes:
            print(f"Training with LR={lr}, BS={bs}")
            train_model(batch_size=bs, learning_rate=lr, epochs=5)

    # 2. Automatically Pick and Export the Winner
    print("\n🏁 All training runs complete. Exporting best model...")
    export_best_model()

if __name__ == "__main__":
    main()