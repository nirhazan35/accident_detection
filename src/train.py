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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

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
        self.val_loss_min = np.Inf
        
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

def train_model(config):
    """
    Train the accident detection model
    
    Args:
        config (dict): Configuration parameters
    """
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'plots'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        accident_dir=config['accident_dir'],
        non_accident_dir=config['non_accident_dir'],
        batch_size=config['batch_size'],
        num_frames=config['num_frames'],
        frame_interval=config['frame_interval'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        num_workers=config['num_workers']
    )
    
    # Get model
    model = get_model(config)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        verbose=True,
        path=os.path.join(config['output_dir'], 'checkpoints', 'best_model.pt')
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    start_time = time.time()
    for epoch in range(config['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for videos, labels in train_bar:
            videos, labels = videos.to(device), labels.to(device).float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * videos.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels.unsqueeze(1)).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}"
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Valid]")
            for videos, labels in val_bar:
                videos, labels = videos.to(device), labels.to(device).float()
                
                # Forward pass
                outputs = model(videos)
                loss = criterion(outputs, labels.unsqueeze(1))
                
                # Update statistics
                val_loss += loss.item() * videos.size(0)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels.unsqueeze(1)).sum().item()
                val_total += labels.size(0)
                
                # Store predictions and targets for metrics
                val_preds.extend(outputs.cpu().numpy())
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
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Calculate additional metrics
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        val_targets = np.array(val_targets)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_targets, val_preds_binary, average='binary'
        )
        
        try:
            auc = roc_auc_score(val_targets, val_preds)
        except:
            auc = 0.0
        
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Early stopping
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
                'val_loss': val_loss
            }, checkpoint_path)
    
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
    """
    Plot training history
    
    Args:
        history (dict): Training history
        output_dir (str): Directory to save plots
    """
    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
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