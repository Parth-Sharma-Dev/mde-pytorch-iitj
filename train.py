import torch
import torch.optim as optim
import os
import time

from models.depth_model import MonocularDepthModel
from utils.dataset import get_train_dataloader
from utils.loss import CustomDepthLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    model = MonocularDepthModel().to(device)
    criterion = CustomDepthLoss(edge_weight=0.1).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    print("Loading dataset...")
    train_loader = get_train_dataloader(
        rgb_dir="./data/nyu_depth_v2/train/rgb",
        depth_dir="./data/nyu_depth_v2/train/depth",
        batch_size=8,       
        num_workers=2       
    )
    print(f"Found {len(train_loader.dataset)} training images.")

    start_epoch = 0
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")
    else:
        print("Starting fresh training run.")
    
    total_epochs = 20 
    model.train()
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        
        for batch_idx, (rgb, target_depth) in enumerate(train_loader):
            rgb = rgb.to(device)
            target_depth = target_depth.to(device)
            
            optimizer.zero_grad()
            
            pred_depth = model(rgb)
            loss = criterion(pred_depth, target_depth)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{total_epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        duration = time.time() - epoch_start
        
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f} | Time: {duration:.2f}s")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        print("Checkpoint saved.\n")

if __name__ == "__main__":
    main()