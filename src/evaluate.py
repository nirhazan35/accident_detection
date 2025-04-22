#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
)
from tqdm import tqdm
import time

from model import get_model
from data_processor import prepare_dataloaders

def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate model on a dataset
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation
        device (torch.device): Device to run evaluation on
        threshold (float): Classification threshold
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    inference_times = []
    
    # Evaluate model
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Evaluating"):
            videos = videos.to(device)
            labels = labels.numpy()
            
            # Measure inference time
            start_time = time.time()
            outputs = model(videos)
            inference_time = time.time() - start_time
            
            # Record inference time per sample
            inference_times.extend([inference_time / videos.size(0)] * videos.size(0))
            
            # Convert outputs to numpy
            predictions = outputs.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute binary predictions using threshold
    binary_predictions = (all_predictions >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, binary_predictions)
    precision = precision_score(all_labels, binary_predictions)
    recall = recall_score(all_labels, binary_predictions)
    f1 = f1_score(all_labels, binary_predictions)
    
    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except:
        auc = 0.0
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, binary_predictions)
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics with optimal threshold
    optimal_predictions = (all_predictions >= optimal_threshold).astype(int)
    optimal_accuracy = accuracy_score(all_labels, optimal_predictions)
    optimal_precision = precision_score(all_labels, optimal_predictions)
    optimal_recall = recall_score(all_labels, optimal_predictions)
    optimal_f1 = f1_score(all_labels, optimal_predictions)
    
    # Return metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "avg_inference_time_ms": float(avg_inference_time),
        "threshold": float(threshold),
        "optimal_threshold": float(optimal_threshold),
        "optimal_accuracy": float(optimal_accuracy),
        "optimal_precision": float(optimal_precision),
        "optimal_recall": float(optimal_recall),
        "optimal_f1": float(optimal_f1),
    }
    
    return metrics, all_predictions, all_labels

def plot_confusion_matrix(cm, save_path):
    """
    Plot confusion matrix
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        save_path (str): Path to save plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    classes = ['No Accident', 'Accident']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text values
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(labels, predictions, save_path):
    """
    Plot ROC curve
    
    Args:
        labels (numpy.ndarray): Ground truth labels
        predictions (numpy.ndarray): Model predictions
        save_path (str): Path to save plot
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # Mark optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
             label=f'Optimal Threshold = {thresholds[optimal_idx]:.3f}')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(labels, predictions, save_path):
    """
    Plot precision-recall curve
    
    Args:
        labels (numpy.ndarray): Ground truth labels
        predictions (numpy.ndarray): Model predictions
        save_path (str): Path to save plot
    """
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    
    # Calculate F1 scores for each threshold
    f1_scores = []
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1 = 0
        f1_scores.append(f1)
    
    # Find optimal threshold (max F1)
    if len(thresholds) > 0:
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point (threshold=0)
        optimal_threshold = thresholds[optimal_idx]
        
        # Mark optimal threshold
        plt.plot(recall[optimal_idx], precision[optimal_idx], 'ro',
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def evaluate(config_path, model_path, output_dir, threshold=0.5):
    """
    Evaluate model and save results
    
    Args:
        config_path (str): Path to configuration file
        model_path (str): Path to model checkpoint
        output_dir (str): Directory to save evaluation results
        threshold (float): Classification threshold
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataloaders
    _, _, test_loader = prepare_dataloaders(
        accident_dir=config['data']['accident_dir'],
        non_accident_dir=config['data']['non_accident_dir'],
        batch_size=config['training'].get('batch_size', 8),
        num_frames=config['data'].get('num_frames', 32),
        frame_interval=config['data'].get('frame_interval', 4),
        train_ratio=config['data'].get('train_ratio', 0.7),
        val_ratio=config['data'].get('val_ratio', 0.15),
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Load model
    model = get_model(config['model'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate model
    metrics, predictions, labels = evaluate_model(model, test_loader, device, threshold)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Average Inference Time: {metrics['avg_inference_time_ms']:.4f} ms")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"With Optimal Threshold:")
    print(f"  Accuracy: {metrics['optimal_accuracy']:.4f}")
    print(f"  Precision: {metrics['optimal_precision']:.4f}")
    print(f"  Recall: {metrics['optimal_recall']:.4f}")
    print(f"  F1 Score: {metrics['optimal_f1']:.4f}")
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Convert confusion matrix to numpy array
    cm = np.array(metrics['confusion_matrix'])
    
    # Plot and save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)
    
    # Plot and save ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(labels, predictions, roc_path)
    
    # Plot and save precision-recall curve
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(labels, predictions, pr_path)
    
    print(f"\nEvaluation results saved to {output_dir}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate accident detection model')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    args = parser.parse_args()
    
    evaluate(args.config, args.model, args.output, args.threshold)

if __name__ == '__main__':
    main() 