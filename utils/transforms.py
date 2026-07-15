import random
from torchvision import transforms
import torchvision.transforms.functional as TF

def joint_transform_with_augmentations(rgb, depth):
    rgb = TF.resize(rgb, (256, 320))
    depth = TF.resize(depth, (256, 320), interpolation=transforms.InterpolationMode.NEAREST)
    
    # randomly flip the images horizontally
    if random.random() > 0.5:
        rgb = TF.hflip(rgb)
        depth = TF.hflip(depth)
        
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    rgb = color_jitter(rgb)
    
    rgb = TF.to_tensor(rgb)
    depth = TF.to_tensor(depth)
    
    rgb = TF.normalize(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return rgb, depth