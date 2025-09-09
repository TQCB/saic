from typing import Tuple
from pydantic import BaseModel, Field, validator

class TrainingConfig(BaseModel):
    """
    A Pydantic model for validating and managing machine learning training configuration.
    """
    # Model Config
    model_name: str = 'saic'
    n: int = Field(128, gt=0, description="Model parameter N, must be a positive integer.")
    m: int = Field(192, gt=0, description="Model parameter M, must be a positive integer.")
    z_alphabet_size: int = Field(201, gt=0, description="Alphabet size for Z, must be positive.")

    # Train Config
    im_size: Tuple[int, int] = (64, 64)
    train_batch_size: int = Field(8, gt=0, description="Training batch size, must be positive.")
    val_batch_size: int = Field(64, gt=0, description="Validation batch size, must be positive.")
    lr: float = Field(1e-3, gt=0, description="Learning rate, must be a positive float.")
    loss_lmbda: float = Field(0.01, gt=0, description='Lambda coefficient used in the loss as a weight for the distortion component')
    loss_fg_weight: float = Field(10, gt=0, description='Foreground coeffficient used in the loss as a weight for the importance of the foreground component.')
    schedule_patience: int = Field(5, ge=0, description="Scheduler patience, must be >= 0.")
    grad_clip: bool = True
    epochs: int = Field(1, gt=0, description="Number of training epochs, must be positive.")
    checkpoint: bool = False
    update_interval: int = Field(100, gt=0, description="Update interval, must be positive.")

    # Custom Validator for im_size
    @validator('im_size')
    def image_dimensions_must_be_positive(cls, v):
        """Validates that both dimensions of im_size are positive."""
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive, but got ({width}, {height})")
        return v

    class Config:
        """Pydantic model configuration."""
        from_attributes = True
        frozen = True