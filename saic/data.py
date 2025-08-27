import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

def get_transforms(train=True, crop_size=(256, 256)):
    """Define the set of transformations for training or validation."""
    if train:
        # For training, use random augmentations
        return transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        # For validation, use a deterministic center crop
        return transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
        ])

class COCOWithMasksDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get a sorted list of image filenames
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # --- Load Image and Mask ---
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Load as grayscale

        # --- Apply Identical Transformations ---
        # Convert to tensor first
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image)
        mask_tensor = to_tensor(mask)
        
        if self.transform:
            # To apply the same random crop/flip, we stack them temporarily,
            # apply the transform, and then unstack.
            stacked = torch.cat([image_tensor, mask_tensor], dim=0)
            stacked_transformed = self.transform(stacked)
            image_tensor, mask_tensor = torch.split(stacked_transformed, [3, 1], dim=0)

        # Normalize the image (values used by many ImageNet models)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        
        return image_tensor, mask_tensor