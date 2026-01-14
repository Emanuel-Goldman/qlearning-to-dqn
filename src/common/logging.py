"""TensorBoard logging utilities."""

from typing import Optional
from torch.utils.tensorboard import SummaryWriter
import os


class TensorBoardLogger:
    """Helper class for TensorBoard logging."""
    
    def __init__(self, log_dir: str, comment: Optional[str] = None):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            comment: Optional comment to append to log directory name
        """
        if comment:
            log_dir = os.path.join(log_dir, comment)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a scalar value.
        
        Args:
            tag: Tag name for the scalar
            value: Scalar value to log
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def increment_step(self) -> None:
        """Increment the global step counter."""
        self.global_step += 1
    
    def set_step(self, step: int) -> None:
        """
        Set the global step counter.
        
        Args:
            step: Step value to set
        """
        self.global_step = step
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()

