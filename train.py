import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import UNet
from dataset import ISICDataset

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def train_model(
    model, 
    device, 
    image_dir, 
    mask_dir, 
    train_image_ids,
    save_dir,
    epochs=25, 
    batch_size=8, 
    lr=1e-4, 
    val_percent=0.1
):
    # 1. Create dataset
    # all_image_ids = [f.replace('.jpg', '') for f in os.listdir(image_dir) if f.endswith('.jpg')]
    train_ids, val_ids = train_test_split(train_image_ids, test_size=val_percent, random_state=42)

    train_dataset = ISICDataset(image_dir, mask_dir, image_ids=train_ids)
    val_dataset = ISICDataset(image_dir, mask_dir, image_ids=val_ids)

    # 2. Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    # 3. Set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)

    # 4. Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)

                masks_pred = model(images)
                
                bce = criterion(masks_pred, true_masks)
                dice = dice_loss(masks_pred, true_masks)
                loss = bce + dice

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks) + dice_loss(masks_pred, true_masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("Saved new best model.")
            
    print("Training finished.")
