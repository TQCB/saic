import torch
import os

class Checkpointer:
    """
    Manages saving and loading model checkpoints
    
    Saves the checkpoint for the current epoch and also keeps track of the
    best model based on validation loss, saving it separately.
    """
    def __init__(self, checkpoint_dir, model_name="model"):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.best_val_loss = float('inf')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint manager initialized. Checkpoints will be saved in '{self.checkpoint_dir}'.")

    def save(self, epoch, model, optimizer, scheduler, val_loss):
        # determine if this is the best model so far
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        # create the state_dict
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'scheduler_state_dict': scheduler.state_dict(),
        }

        # save the latest checkpoint
        latest_filename = f"{self.model_name}_latest.pth"
        latest_filepath = os.path.join(self.checkpoint_dir, latest_filename)
        torch.save(state, latest_filepath)

        # 4. If it's the best model, save it to a separate file
        if is_best:
            best_filename = f"{self.model_name}_best.pth"
            best_filepath = os.path.join(self.checkpoint_dir, best_filename)
            torch.save(state, best_filepath)
            
    def load(self, model, optimizer, scheduler, load_best=True):
        """
        load a checkpoint into the model, optimizer, and scheduler.

        Args:
            model, optimizer, scheduler: The objects to load the state into.
            load_best (bool): If True, loads the best model. Otherwise, loads the latest.

        Returns:
            int: The epoch number to resume training from. Returns 0 if no checkpoint is found.
        """
        filename = f"{self.model_name}_{'best' if load_best else 'latest'}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}. Starting training from scratch.")
            return 0

        # load the checkpoint dictionary
        checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # load the states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # load metadata
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"Loaded checkpoint from {filepath}. Resuming training from epoch {start_epoch}.")
        print(f"Best validation loss so far was {self.best_val_loss:.4f}.")
        
        return start_epoch