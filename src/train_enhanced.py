#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import logging
from datetime import datetime

from train import train_model, evaluate_model
from model import get_model
from data_processor import prepare_dataloaders

def load_config(config_path, model_name):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Combine common config with model-specific config
    config = {**configs['common'], **configs[model_name]}
    return config

def setup_logger(save_dir):
    """Set up logger for training process"""
    os.makedirs(save_dir, exist_ok=True)
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

def main():
    """Main function to train enhanced models"""
    parser = argparse.ArgumentParser(description='Train enhanced accident detection models')
    parser.add_argument('--config', type=str, default='configs/enhanced_models.yaml', 
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['spatial_channel_attention', 'efficientnetv2', 
                                 'convnext', 'slowfast_inspired', 'x3d'],
                        help='Model architecture to train')
    parser.add_argument('--data_dir', type=str, help='Data directory (overrides config)')
    parser.add_argument('--save_dir', type=str, help='Save directory (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate', action='store_true', help='Only run evaluation')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for evaluation')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config, args.model)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.save_dir:
        config['save_dir'] = args.save_dir
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    
    # Create model save directory with model name
    model_save_dir = os.path.join(config['save_dir'], args.model)
    config['save_dir'] = model_save_dir
    
    # Set up logger
    logger = setup_logger(model_save_dir)
    logger.info(f"Training model: {args.model}")
    logger.info(f"Config: {config}")
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = get_model(config)
    model.to(device)
    
    # Define data paths
    accident_dir = os.path.join(config['data_dir'], 'processed/accidents')
    non_accident_dir = os.path.join(config['data_dir'], 'processed/non_accidents')
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        accident_dir=accident_dir,
        non_accident_dir=non_accident_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        balance_classes=True
    )
    
    if args.evaluate:
        # Evaluate model
        assert args.checkpoint, "Checkpoint path must be provided for evaluation"
        logger.info(f"Evaluating model from checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        
        metrics, _, _ = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=model_save_dir
        )
        
        logger.info(f"Evaluation results:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")
    else:
        # Train model
        trained_model, history = train_model(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            resume_path=args.resume
        )
        
        # Save final model
        final_model_path = os.path.join(model_save_dir, 'final_model.pt')
        torch.save(trained_model.state_dict(), final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        # Evaluate model on test set
        logger.info("Evaluating model on test set...")
        metrics, _, _ = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            device=device,
            output_dir=model_save_dir
        )
        
        logger.info(f"Test results:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")

if __name__ == '__main__':
    main() 