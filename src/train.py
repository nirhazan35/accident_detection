import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, average_precision_score
import gc
from pathlib import Path

from data_processor import prepare_dataloaders, VideoDataset
from model import LSTMTransformerModel, get_model

def setup_logger(save_dir):
    """Set up logger for training process"""
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Run one training epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device).float()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(videos)
        loss = criterion(outputs, labels.unsqueeze(1))
        
        # Backward + optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.view(-1) == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Run validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc='Validation'):
            videos = videos.to(device)
            labels = labels.to(device).float()
            
            outputs = model(videos)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.view(-1) == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=7, min_delta=0, verbose=True, path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with class weights
    """
    def __init__(self, pos_weight=1.0, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, outputs, targets):
        # Weight matrix: positive samples get more weight
        weight = torch.ones_like(targets)
        weight[targets == 1] = self.pos_weight
        
        # Binary cross entropy
        loss = nn.functional.binary_cross_entropy(
            outputs, 
            targets, 
            weight=weight, 
            reduction=self.reduction
        )
        
        return loss

def mixup_data(x, y, alpha=0.2):
    """
    Performs mixup augmentation on the batch.
    
    Args:
        x: input features [batch_size, ...]
        y: input labels [batch_size, ...]
        alpha: mixup interpolation coefficient
        
    Returns:
        mixed_x, mixed_y, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y, lam

# For Windows path handling
def ensure_dir_exists(path):
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)
    return path

# Add function to clear GPU memory
def clear_gpu_memory():
    """Clear GPU memory cache to prevent memory leaks"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Function to configure CUDA for optimal performance
def configure_cuda():
    """Configure CUDA for optimal performance"""
    if torch.cuda.is_available():
        # Set device to current device
        torch.cuda.set_device(torch.cuda.current_device())
        
        # Enable TF32 precision on Ampere GPUs (faster than FP32, almost as accurate)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set other CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Find optimal algorithms
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        
        # Print CUDA info for debugging
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA Device: {device_name}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Check if we can use Tensor Cores for mixed precision
        can_use_amp = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
        if can_use_amp:
            print("Mixed precision training available")
        else:
            print("Mixed precision training not available")
        
        return can_use_amp
    return False

def train_model(config):
    """
    Train the accident detection model
    
    Args:
        config (dict): Configuration parameters
    """
    # Create output directories - use Path for Windows compatibility
    output_dir = Path(config['output_dir'])
    ensure_dir_exists(output_dir)
    ensure_dir_exists(output_dir / 'checkpoints')
    ensure_dir_exists(output_dir / 'plots')
    
    # Configure CUDA
    can_use_amp = configure_cuda()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if can_use_amp else None
    
    # For performance profiling
    data_loading_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        accident_dir=config['data']['accident_dir'],
        non_accident_dir=config['data']['non_accident_dir'],
        batch_size=config['training']['batch_size'],
        num_frames=config['data']['num_frames'],
        frame_interval=config['data']['frame_interval'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        num_workers=config['training']['num_workers'],
        balance_classes=config.get('training', {}).get('balance_classes', True)
    )
    
    # Get model
    model = get_model(config['model'])
    model = model.to(device)
    
    # Calculate class weights for loss function
    if config.get('training', {}).get('use_weighted_loss', True):
        # If we know the exact class distribution
        num_accident = len(os.listdir(config['data']['accident_dir']))
        num_non_accident = len(os.listdir(config['data']['non_accident_dir']))
        pos_weight = num_non_accident / num_accident if num_accident > 0 else 1.0
        print(f"Using weighted loss with positive weight: {pos_weight:.2f}")
        criterion = WeightedBCELoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        verbose=True,
        path=str(Path(config['output_dir']) / 'checkpoints' / 'best_model.pt')
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_auc': [],
        'val_ap': []  # Average Precision
    }
    
    # Use mixup augmentation
    use_mixup = config.get('training', {}).get('use_mixup', True)
    mixup_alpha = config.get('training', {}).get('mixup_alpha', 0.2)
    
    # Training loop
    start_time = time.time()
    for epoch in range(config['training']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Train]")
        for videos, labels in train_bar:
            # Measure data loading time
            data_load_end = time.time()
            
            videos, labels = videos.to(device), labels.to(device).float()
            
            # Apply mixup augmentation
            if use_mixup and epoch > 5:  # Start mixup after 5 epochs of regular training
                videos, labels, _ = mixup_data(videos, labels.unsqueeze(1), alpha=mixup_alpha)
                labels = labels.squeeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            forward_start = time.time()
            
            # Use mixed precision if available
            if can_use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(videos)
                    # Calculate loss
                    if use_mixup and epoch > 5:
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels.unsqueeze(1))
            else:
                outputs = model(videos)
                # Calculate loss
                if use_mixup and epoch > 5:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels.unsqueeze(1))
                    
            forward_end = time.time()
            
            # Backward pass
            backward_start = time.time()
            if can_use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            backward_end = time.time()
            
            # Update timing statistics
            forward_time += forward_end - forward_start
            backward_time += backward_end - backward_start
            
            # Update statistics
            train_loss += loss.item() * videos.size(0)
            predicted = (outputs > 0.5).float()
            
            if use_mixup and epoch > 5:
                # For mixup, use the dominant class for accuracy calculation
                rounded_labels = (labels > 0.5).float()
                train_correct += (predicted == rounded_labels).sum().item()
            else:
                train_correct += (predicted == labels.unsqueeze(1)).sum().item()
                
            train_total += labels.size(0)
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}"
            })
            
            # Record data loading time for next iteration
            data_loading_time += data_load_end - time.time()
        
        # Print timing info after each epoch
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"\nTiming Info (Epoch {epoch+1}):")
            print(f"  Data loading time: {data_loading_time:.2f}s")
            print(f"  Forward pass time: {forward_time:.2f}s")
            print(f"  Backward pass time: {backward_time:.2f}s")
            print(f"  Total compute time: {forward_time + backward_time:.2f}s")
            print(f"  Ratio - Data loading : Computing = {data_loading_time / (forward_time + backward_time):.2f}")
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_preds_raw = []  # For ROC-AUC and PR curves
        val_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Valid]")
            for videos, labels in val_bar:
                videos, labels = videos.to(device), labels.to(device).float()
                
                # Forward pass with mixed precision
                if can_use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(videos)
                        loss = criterion(outputs, labels.unsqueeze(1))
                else:
                    outputs = model(videos)
                    loss = criterion(outputs, labels.unsqueeze(1))
                
                # Update statistics
                val_loss += loss.item() * videos.size(0)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels.unsqueeze(1)).sum().item()
                val_total += labels.size(0)
                
                # Store predictions and targets for metrics
                val_preds.extend(predicted.cpu().numpy())
                val_preds_raw.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{val_correct/val_total:.4f}"
                })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate evaluation metrics
        val_preds_binary = np.array(val_preds).reshape(-1) > 0.5
        val_preds_raw = np.array(val_preds_raw).reshape(-1)
        val_targets = np.array(val_targets)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_targets, val_preds_binary, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(val_targets, val_preds_raw)
        except:
            auc = 0.0
            
        try:
            ap = average_precision_score(val_targets, val_preds_raw)
        except:
            ap = 0.0
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['val_auc'].append(auc)
        history['val_ap'].append(ap)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
        
        # Early stopping - now monitoring F1 score instead of just loss
        if config.get('training', {}).get('monitor_f1', True):
            early_stopping(-f1, model)  # Negative because we want to maximize F1
        else:
            early_stopping(val_loss, model)
            
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['output_dir'], 
                'checkpoints', 
                f'model_epoch_{epoch+1}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1': f1,
                'val_auc': auc
            }, checkpoint_path)
        
        # Clean up GPU memory at the end of each epoch
        clear_gpu_memory()
    
    # Training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    # Plot training history
    plot_history(history, config['output_dir'])
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'checkpoints', 'best_model.pt')))
    
    # Test the model
    test_metrics = evaluate_model(model, test_loader, device, config['output_dir'])
    
    return model, history, test_metrics

def evaluate_model(model, test_loader, device, output_dir):
    """
    Evaluate the model on the test set
    
    Args:
        model: The trained model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        output_dir: Directory to save results
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    criterion = nn.BCELoss()
    
    test_loss = 0.0
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Testing"):
            videos, labels = videos.to(device), labels.to(device).float()
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Update statistics
            test_loss += loss.item() * videos.size(0)
            
            # Store predictions and targets
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    
    # Convert to numpy arrays
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_preds_binary = (test_preds > 0.5).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets, test_preds_binary, average='binary'
    )
    
    accuracy = (test_preds_binary.flatten() == test_targets).mean()
    
    try:
        auc = roc_auc_score(test_targets, test_preds)
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(test_targets, test_preds_binary)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Non-Accident', 'Accident']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix.png'))
    
    # Print results
    print(f"Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Save metrics
    metrics = {
        'loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def plot_history(history, output_dir):
    """Plot training history metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot F1, Precision, Recall
    plt.subplot(2, 2, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Precision, Recall, F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Plot AUC and AP
    plt.subplot(2, 2, 4)
    plt.plot(history['val_auc'], label='ROC AUC')
    plt.plot(history['val_ap'], label='Average Precision')
    plt.title('ROC AUC and Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'training_history.png'))
    plt.close()

if __name__ == '__main__':
    import argparse
    import json
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train accident detection model')
    parser.add_argument('--config', type=str, default='config.json', 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Train model
    model, history, test_metrics = train_model(config)
    
    # Save test metrics
    with open(os.path.join(config['output_dir'], 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4) 