import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .transforms import joint_transform_with_augmentations

class NYUDepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        
        self.image_filenames = sorted(os.listdir(rgb_dir))
        self.depth_filenames = sorted(os.listdir(depth_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.image_filenames[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_filenames[idx])
        
        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_map = Image.open(depth_path) 
        
        if self.transform:
            rgb_image, depth_map = self.transform(rgb_image, depth_map)
            
        return rgb_image, depth_map

def get_train_dataloader(rgb_dir, depth_dir, batch_size=8, num_workers=2):
    dataset = NYUDepthDataset(rgb_dir, depth_dir, transform=joint_transform_with_augmentations)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )