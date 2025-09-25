"""Custom training callbacks."""

import torch
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, List


class TensorBoardCallback(TrainerCallback):
    """Custom callback to log additional metrics to TensorBoard."""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics during training."""
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics."""
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key.startswith("eval_"):
                    self.writer.add_scalar(f"eval/{key[5:]}", value, state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Close TensorBoard writer."""
        self.writer.close()


class GradientClippingCallback(TrainerCallback):
    """Custom callback to apply additional gradient clipping and monitoring."""
    
    def __init__(self, max_grad_norm: float = 1.0, clip_threshold: float = 10.0):
        """
        Initialize gradient clipping callback.
        
        Args:
            max_grad_norm: Maximum gradient norm for clipping
            clip_threshold: Threshold above which to apply clipping
        """
        self.max_grad_norm = max_grad_norm
        self.clip_threshold = clip_threshold
        self.gradient_norms: List[float] = []
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Apply gradient clipping after each step."""
        if model is not None:
            # Calculate total gradient norm
            total_norm = 0.0
            param_count = 0
            has_nan = False
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Check for NaN gradients
                    if torch.isnan(param.grad).any():
                        print(f"⚠️  WARNING: NaN gradient detected in {name}")
                        has_nan = True
                        # Zero out NaN gradients
                        param.grad.data = torch.zeros_like(param.grad.data)
                        continue
                    
                    # Calculate norm
                    param_norm = param.grad.data.norm(2)
                    if not torch.isnan(param_norm) and not torch.isinf(param_norm):
                        total_norm += param_norm.item() ** 2
                        param_count += 1
            
            if param_count > 0 and not has_nan:
                total_norm = total_norm ** 0.5
                self.gradient_norms.append(total_norm)
                
                # Apply gradient clipping if norm is too large
                if total_norm > self.clip_threshold:
                    print(f"⚠️  WARNING: Large gradient norm detected: {total_norm:.2f}, applying clipping...")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                
                # Log gradient norm periodically
                if state.global_step % 10 == 0:
                    print(f"Step {state.global_step}: Gradient norm = {total_norm:.4f}")
            elif has_nan:
                print(f"⚠️  WARNING: NaN gradients detected, skipping gradient norm calculation")
                self.gradient_norms.append(float('nan'))
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add gradient norm to logs."""
        if logs and self.gradient_norms:
            recent_norm = self.gradient_norms[-1] if self.gradient_norms else 0.0
            logs['gradient_norm'] = recent_norm
    
    def get_stats(self) -> dict:
        """
        Get gradient statistics.
        
        Returns:
            Dictionary with gradient statistics
        """
        if not self.gradient_norms:
            return {}
            
        valid_norms = [n for n in self.gradient_norms if not torch.isnan(torch.tensor(n))]
        
        if not valid_norms:
            return {'all_nan': True}
            
        return {
            'mean_norm': sum(valid_norms) / len(valid_norms),
            'max_norm': max(valid_norms),
            'min_norm': min(valid_norms),
            'nan_count': len(self.gradient_norms) - len(valid_norms),
            'total_steps': len(self.gradient_norms)
        }


class EarlyStoppingCallback(TrainerCallback):
    """Callback for early stopping based on validation metrics."""
    
    def __init__(self,
                 patience: int = 3,
                 metric: str = "eval_loss",
                 greater_is_better: bool = False,
                 threshold: Optional[float] = None):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of evaluations to wait before stopping
            metric: Metric to monitor
            greater_is_better: Whether higher values are better
            threshold: Optional threshold for the metric
        """
        self.patience = patience
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.threshold = threshold
        self.best_value = None
        self.patience_counter = 0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check if training should stop."""
        if metrics and self.metric in metrics:
            current_value = metrics[self.metric]
            
            # Check threshold if specified
            if self.threshold is not None:
                if self.greater_is_better and current_value >= self.threshold:
                    print(f"Metric {self.metric} reached threshold {self.threshold}")
                    control.should_training_stop = True
                    return control
                elif not self.greater_is_better and current_value <= self.threshold:
                    print(f"Metric {self.metric} reached threshold {self.threshold}")
                    control.should_training_stop = True
                    return control
            
            # Check for improvement
            if self.best_value is None:
                self.best_value = current_value
            else:
                improved = False
                if self.greater_is_better and current_value > self.best_value:
                    improved = True
                elif not self.greater_is_better and current_value < self.best_value:
                    improved = True
                    
                if improved:
                    self.best_value = current_value
                    self.patience_counter = 0
                    print(f"Improvement detected: {self.metric} = {current_value:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"No improvement for {self.patience_counter} evaluations")
                    
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {self.patience} evaluations without improvement")
                        control.should_training_stop = True
                        
        return control
