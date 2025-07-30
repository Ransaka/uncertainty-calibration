import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import UNet
from dataset import ISICDataset
from train import train_model
from uncertainty import generate_uncertainty_maps
from evaluate import tune_lambda_and_evaluate

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = 'data/ISIC2018_Task1-2_Training_Input/'
MASK_DIR = 'data/ISIC2018_Task1_Training_GroundTruth/'
WEIGHTS_SAVE_DIR = 'results/weights/'
UNCERTAINTY_SAVE_DIR = 'results/uncertainty_maps/'
MODEL_PATH = os.path.join(WEIGHTS_SAVE_DIR, 'best_model.pth')

# Training Hyperparameters
EPOCHS = 2 # Set to a higher number for full training
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# Uncertainty Hyperparameters
MC_PASSES = 30 # Number of Monte Carlo passes

# Create validation and test sets
all_ids = [f.replace('.jpg', '') for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
train_val_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_val_ids, test_size=0.125, random_state=42) # 0.125 * 0.8 = 0.1

# --- Main Workflow ---
if __name__ == '__main__':
    
    # --- 1. Train the Model (or skip if already trained) ---
    print("--- Step 1: Model Training ---")
    if not os.path.exists(MODEL_PATH):
        print("Model weights not found. Starting training...")
        model = UNet(n_channels=3, n_classes=1).to(DEVICE)
        train_model(
                    model=model,
                    device=DEVICE,
                    train_image_ids=train_ids,
                    mask_dir=MASK_DIR,
                    image_dir=IMAGE_DIR,
                    save_dir=WEIGHTS_SAVE_DIR,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    lr=LEARNING_RATE
                )
    else:
        print("Found existing model weights. Skipping training.")

    # --- 2. Generate Uncertainty Maps ---
    print("\n--- Step 2: Generating Uncertainty Maps ---")
    # Load the trained model
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()


    # Create dataloaders
    val_dataset = ISICDataset(IMAGE_DIR, MASK_DIR, image_ids=val_ids)
    test_dataset = ISICDataset(IMAGE_DIR, MASK_DIR, image_ids=test_ids)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Generate maps for validation set
    generate_uncertainty_maps(
        model=model,
        dataloader=val_loader,
        device=DEVICE,
        num_passes=MC_PASSES,
        save_dir=os.path.join(UNCERTAINTY_SAVE_DIR, 'validation')
    )
    
    # Generate maps for test set
    generate_uncertainty_maps(
        model=model,
        dataloader=test_loader,
        device=DEVICE,
        num_passes=MC_PASSES,
        save_dir=os.path.join(UNCERTAINTY_SAVE_DIR, 'test')
    )
    
    # --- 3. Tune Lambda and Evaluate ---
    print("\n--- Step 3: Tuning and Final Evaluation ---")
    tune_lambda_and_evaluate(
        val_dir=os.path.join(UNCERTAINTY_SAVE_DIR, 'validation'),
        test_dir=os.path.join(UNCERTAINTY_SAVE_DIR, 'test')
    )
