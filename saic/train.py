import os

import torch
from dotenv import load_dotenv

from utils import plot_progress
from config import TrainingConfig
from checkpoint import Checkpointer
from loss import SpatialRateDistortionLoss
from data import COCOWithMasksDataset, get_transforms
from compression import HyperpriorCheckerboardCompressor

load_dotenv()

# model config
MODEL_NAME = 'saic'
N = 32 # Default is 128
M = 64 # Default is 192
Z_ALPHABET_SIZE = 201

# train config
IM_SIZE = (64, 64)
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 64
LR = 1e-3
SCHEDULE_PATIENCE = 5
GRAD_CLIP = True
EPOCHS = 1
CHECKPOINT = False
UPDATE_INTERVAL = 100

def train(config: TrainingConfig):
    checkpoint_dir = os.environ['CHECKPOINT_DIR']
    checkpointer = Checkpointer(checkpoint_dir, config.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperpriorCheckerboardCompressor(n=N, m=M, z_alphabet_size=Z_ALPHABET_SIZE).to(device)

    criterion = SpatialRateDistortionLoss(lmbda=0.01, foreground_weight=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULE_PATIENCE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model initialized with {total_params:,} trainable parameters.")

    # TRAIN DATA
    train_image_dir = os.environ['TRAIN_COCO_IMAGE_DIR']
    train_mask_dir = os.environ['TRAIN_COCO_MASK_DIR']
    
    train_dataset = COCOWithMasksDataset(
        train_image_dir,
        train_mask_dir,
        transform=get_transforms(crop_size=IM_SIZE)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # VAL DATA
    val_image_dir = os.environ['VAL_COCO_IMAGE_DIR']
    val_mask_dir = os.environ['VAL_COCO_MASK_DIR']

    val_dataset = COCOWithMasksDataset(
        val_image_dir, 
        val_mask_dir, 
        transform=get_transforms(crop_size=IM_SIZE, train=False)
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )

    print("Data initialized.")
    print("Training loop beginning.")

    epochs = EPOCHS
    for epoch in range(epochs):
        model.train()
        for i, (image, mask) in enumerate(train_loader):
            image = image.to(device)
            mask = mask.to(device)

            output_dict = model(image, mask)
            loss = criterion(output_dict, image, mask)

            optimizer.zero_grad()
            loss.backward()

            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if (i+1) % UPDATE_INTERVAL == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{i+1}/{len(train_loader)}] | Train Loss: {loss.item():.4f}")
                plot_progress(
                    checkpoint_dir,
                    epoch,
                    i,
                    output_dict,
                    image,
                    criterion.R_bpp,
                )

        # VALIDATION
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_image, val_mask in val_loader:
                val_image = val_image.to(device)
                val_mask = val_mask.to(device)

                output_dict = model(val_image, val_mask)
                loss = criterion(output_dict, image, mask)
                total_val_loss += loss.item()
                
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}] | Validation Loss: {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)

            if CHECKPOINT:
                checkpointer.save(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    val_loss=avg_val_loss,
                    scheduler=scheduler
                )

if __name__ == '__main__':
    settings = {}

    try:
        validated_config = TrainingConfig(**settings)
        train(validated_config)
        
    except Exception as e:
        print(f"Configuration error: {e}")