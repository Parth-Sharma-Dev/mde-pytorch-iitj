import torch
import numpy as np
from models.depth_model import MonocularDepthModel
from utils.dataset import get_train_dataloader

def calculate_metrics(pred, target):
    valid_mask = target > 0
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    rmse = torch.sqrt(torch.mean((target - pred) ** 2))
    
    thresh = torch.max((target / pred), (pred / target))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    
    return rmse.item(), a1.item(), a2.item(), a3.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing Evaluation...")

    val_loader = get_train_dataloader(
        rgb_dir="./data/nyu_depth_v2/val/rgb",
        depth_dir="./data/nyu_depth_v2/val/depth",
        batch_size=8,
        num_workers=2
    )
    
    model = MonocularDepthModel().to(device)
    checkpoint = torch.load("./checkpoints/latest_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_rmse, total_a1, total_a2, total_a3 = 0, 0, 0, 0
    num_batches = len(val_loader)

    print("Running metrics on validation set...")
    with torch.no_grad():
        for batch_idx, (rgb, target) in enumerate(val_loader):
            rgb, target = rgb.to(device), target.to(device)
            pred = model(rgb)
            
            rmse, a1, a2, a3 = calculate_metrics(pred, target)
            total_rmse += rmse
            total_a1 += a1
            total_a2 += a2
            total_a3 += a3

    print("FINAL EVALUATION METRICS")
    print(f"RMSE:     {total_rmse / num_batches:.4f}")
    print(f"Delta 1:  {total_a1 / num_batches:.4f} (Strict Accuracy)")
    print(f"Delta 2:  {total_a2 / num_batches:.4f}")
    print(f"Delta 3:  {total_a3 / num_batches:.4f} (Loose Accuracy)")
    print("="*30)

if __name__ == "__main__":
    main()