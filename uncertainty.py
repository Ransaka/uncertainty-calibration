import os
import torch
import numpy as np
from tqdm import tqdm
import cv2

from model import enable_dropout

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        # Target for segmentation is the sum of logits for the foreground class
        score = output.sum()
        score.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        
        for i in range(pooled_gradients.size(0)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

def generate_uncertainty_maps(model, dataloader, device, num_passes, save_dir):
    """
    Generates and saves predictive and saliency variance maps.
    """
    model.to(device)
    enable_dropout(model) # IMPORTANT: activate dropout for MC passes
    
    # Initialize Grad-CAM
    # Assuming the target layer is the last conv block in the encoder
    grad_cam = GradCAM(model, model.down4.maxpool_conv[1])

    os.makedirs(save_dir, exist_ok=True)
    
    for batch in tqdm(dataloader, desc="Generating Uncertainty Maps"):
        images = batch['image'].to(device)
        true_masks = batch['mask']
        image_ids = batch['id']

        for i in range(images.size(0)):
            img = images[i].unsqueeze(0)
            img_id = image_ids[i]
            
            # Check if map already exists
            if os.path.exists(os.path.join(save_dir, f"{img_id}.npz")):
                continue

            mc_predictions = []
            mc_saliency_maps = []

            for _ in range(num_passes):
                with torch.no_grad():
                    pred_logit = model(img)
                    pred_prob = torch.sigmoid(pred_logit).squeeze().cpu().numpy()
                
                # Get saliency map (requires gradients)
                saliency_map = grad_cam(img)
                saliency_map = cv2.resize(saliency_map, (img.shape[3], img.shape[2])) # Resize to match output

                mc_predictions.append(pred_prob)
                mc_saliency_maps.append(saliency_map)

            # Stack and compute variance
            pred_variance = np.var(np.stack(mc_predictions, axis=0), axis=0)
            saliency_variance = np.var(np.stack(mc_saliency_maps, axis=0), axis=0)
            
            # Use mean prediction for accuracy calculation later
            mean_prediction = np.mean(np.stack(mc_predictions, axis=0), axis=0)
            
            # Save the maps
            np.savez_compressed(
                os.path.join(save_dir, f"{img_id}.npz"),
                pred_variance=pred_variance,
                saliency_variance=saliency_variance,
                mean_prediction=mean_prediction,
                true_mask=true_masks[i].squeeze().cpu().numpy()
            )
