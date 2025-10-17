import os

import torch
from tqdm import tqdm

from utils import plot_progress
from config import TrainingConfig
from checkpoint import Checkpointer
from loss import SpatialRateDistortionLoss
from data import COCOWithMasksDataset, get_transforms
from compression import HyperpriorCheckerboardCompressor



def train(config: TrainingConfig):
    if config.checkpoint & (not config.validation):
        raise Exception("Checkpointing cannot be enabled without validation.")

    checkpoint_dir = os.environ['CHECKPOINT_DIR']
    checkpointer = Checkpointer(checkpoint_dir, config.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperpriorCheckerboardCompressor(n=config.n, m=config.m, z_alphabet_size=config.z_alphabet_size)
    model = model.to(device, memory_format=torch.channels_last)
    # model = torch.compile(model)

    criterion = SpatialRateDistortionLoss(lmbda=config.loss_lmbda, foreground_weight=config.loss_fg_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.schedule_patience)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model initialized with {total_params:,} trainable parameters on {device} device")

    # TRAIN DATA
    train_image_dir = os.environ['TRAIN_COCO_IMAGE_DIR']
    train_mask_dir = os.environ['TRAIN_COCO_MASK_DIR']
    
    train_dataset = COCOWithMasksDataset(
        train_image_dir,
        train_mask_dir,
        transform=get_transforms(crop_size=config.im_size)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )
    
    # VAL DATA
    val_image_dir = os.environ['VAL_COCO_IMAGE_DIR']
    val_mask_dir = os.environ['VAL_COCO_MASK_DIR']

    val_dataset = COCOWithMasksDataset(
        val_image_dir, 
        val_mask_dir, 
        transform=get_transforms(crop_size=config.im_size, train=False)
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )

    # gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(device.type)

    print("Data initialized.")
    print("Training loop beginning.")

    epochs = config.epochs
    for epoch in tqdm(range(epochs), desc='Epochs'):
        model.train()
        for i, (image, mask) in enumerate(train_loader):
            image = image.to(device, memory_format=torch.channels_last)
            mask = mask.to(device, memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type):
                output_dict = model(image, mask)
                loss = criterion(output_dict, image, mask)

            scaler.scale(loss).backward()

            if config.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            if (i+1) % config.update_interval == 0:
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
        if not config.validation:
            continue

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_image, val_mask in val_loader:
                val_image = val_image.to(device)
                val_mask = val_mask.to(device)

                output_dict = model(val_image, val_mask)
                loss = criterion(output_dict, val_image, val_mask)
                total_val_loss += loss.item()
                
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}] | Validation Loss: {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)

            if config.checkpoint:
                checkpointer.save(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    val_loss=avg_val_loss,
                    scheduler=scheduler
                )

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    print("Configuring model...")
    experiment_settings = dict(
        model_name = 'local_saic',
        im_size = (64, 64),
        lr = 1e-3,
        scheduler_patience = 1,
        epochs = 20,
        train_batch_size = 32,
        checkpoint = False,
        n = int(16),
        m = int(16),
        validation = False,
        update_interval = 100,
    )

    try:
        config = TrainingConfig(**experiment_settings)
        print("Model configured.")
    except Exception as e:
        print(f"Configuration error: {e}")

    train(config)