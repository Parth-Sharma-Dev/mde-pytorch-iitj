import torch
import torchvision.transforms.functional as TF
import cv2
import matplotlib.pyplot as plt
import argparse

from models.depth_model import MonocularDepthModel

def load_and_preprocess_image(image_path, device):
    print(f"Loading image from {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    original_img = img.copy()
    
    from PIL import Image
    img = Image.fromarray(img)
    
    img = TF.resize(img, (256, 320))
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    img = img.unsqueeze(0).to(device)
    
    return img, original_img

def main(image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MonocularDepthModel().to(device)
    
    print(f"Loading trained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval() 
    
    input_tensor, original_img = load_and_preprocess_image(image_path, device)
    
    print("Generating depth map...")
    with torch.no_grad(): 
        pred_depth = model(input_tensor)
        
    depth_map = pred_depth.squeeze().cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original RGB Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted Depth Map")
    plt.imshow(depth_map, cmap='inferno_r')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Monocular Depth Model")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/latest_model.pth", help="Path to model weights")
    
    args = parser.parse_args()
    main(args.image, args.checkpoint)