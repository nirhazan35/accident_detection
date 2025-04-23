#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess
import logging
import argparse
import torch
import yaml
import shutil
from datetime import datetime
from pathlib import Path

# Configure logging
def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"batch_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def optimize_cuda_settings():
    """Set optimal CUDA settings for training"""
    if torch.cuda.is_available():
        # Use TF32 precision on Ampere GPUs for faster training with minimal accuracy loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set to benchmark mode for optimal performance when input sizes don't change
        torch.backends.cudnn.benchmark = True
        
        # Don't use deterministic algorithms (slower but set to True if reproducibility is critical)
        torch.backends.cudnn.deterministic = False
        
        # Empty cache before starting
        torch.cuda.empty_cache()
        
        logger.info(f"CUDA optimized. Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available. Training will be done on CPU.")

def update_config_for_model(model_name):
    """Update batch size based on model complexity for 6GB VRAM"""
    # Load the config
    config_path = 'configs/enhanced_models.yaml'
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Adjust batch sizes and parameters based on model and VRAM
    if model_name == 'efficientnetv2' or model_name == 'convnext':
        configs['common']['batch_size'] = 2  # Larger models need smaller batches
    elif model_name == 'x3d':
        configs['common']['batch_size'] = 2  # 3D models need even smaller batches
    else:
        configs['common']['batch_size'] = 4  # Standard batch size for simpler models
    
    # Adjust workers based on batch size to avoid memory issues
    configs['common']['num_workers'] = min(4, configs['common']['batch_size'] * 2)
    
    # Save configurations for this run
    run_configs_dir = os.path.join("logs", "configs")
    os.makedirs(run_configs_dir, exist_ok=True)
    
    model_config_path = os.path.join(run_configs_dir, f"{model_name}_config.yaml")
    with open(model_config_path, 'w') as f:
        yaml.dump({**configs['common'], **configs[model_name]}, f, default_flow_style=False)
    
    return configs['common']['batch_size']

def run_model(model_name):
    """Run training for a specific model with optimized settings"""
    logger.info(f"======== Starting {model_name} training ========")
    
    # Update config and get batch size
    batch_size = update_config_for_model(model_name)
    
    # Define save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"saved_models/enhanced/{model_name}_{timestamp}"
    
    # Define data directory
    data_dir = "data"  # Use the root data directory
    
    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "src/train_enhanced.py",
        "--model", model_name,
        "--save_dir", save_dir,
        "--data_dir", data_dir,
        "--batch_size", str(batch_size)
    ]
    
    # Log command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute training
    start_time = time.time()
    try:
        # Run the process and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Create a dedicated log file for this model
        model_log_path = os.path.join("logs", f"{model_name}_{timestamp}.txt")
        with open(model_log_path, 'w') as model_log_file:
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')  # Print to console
                model_log_file.write(line)  # Write to model-specific log file
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"{model_name} training completed successfully in {elapsed_time:.2f} seconds")
            
            # Copy final model with a more descriptive name
            final_model_path = os.path.join(save_dir, "final_model.pt")
            descriptive_name = f"{model_name}_final_{timestamp}.pt"
            descriptive_path = os.path.join("saved_models", descriptive_name)
            
            if os.path.exists(final_model_path):
                shutil.copy2(final_model_path, descriptive_path)
                logger.info(f"Saved model to {descriptive_path}")
            
            return True
        else:
            logger.error(f"{model_name} training failed with return code {return_code}")
            return False
            
    except Exception as e:
        logger.exception(f"Error running {model_name} training: {str(e)}")
        return False

def cleanup():
    """Clean up temporary files and CUDA cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Remove any temporary files if needed
    temp_files = []  # Add paths to any temp files here
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except:
                pass

def create_summary(results):
    """Create a summary of all training runs"""
    summary_path = os.path.join("logs", f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(summary_path, 'w') as f:
        f.write("===== ACCIDENT DETECTION MODEL TRAINING SUMMARY =====\n\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models trained: {len(results)}\n\n")
        
        for model_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{model_name}: {status}\n")
        
        f.write("\nCheck individual log files for detailed metrics.\n")
    
    logger.info(f"Summary created at {summary_path}")

def main():
    # Ensure directories exist
    os.makedirs("saved_models/enhanced", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Check data paths - should match what train_enhanced.py expects
    data_root = "data"
    data_accident_path = os.path.join(data_root, "processed/accidents")
    data_non_accident_path = os.path.join(data_root, "processed/non_accidents")
    
    if not os.path.exists(data_accident_path) or not os.path.exists(data_non_accident_path):
        logger.error(f"Data directories not found. Please ensure {data_accident_path} and {data_non_accident_path} exist.")
        return
    
    logger.info(f"Found accident videos: {len([f for f in os.listdir(data_accident_path) if f.endswith(('.mp4', '.avi', '.mov'))])}")
    logger.info(f"Found non-accident videos: {len([f for f in os.listdir(data_non_accident_path) if f.endswith(('.mp4', '.avi', '.mov'))])}")
    
    # Optimize CUDA settings
    optimize_cuda_settings()
    
    # Models to train
    models = [
        'spatial_channel_attention',
        'efficientnetv2',
        'convnext',
        'slowfast_inspired',
        'x3d'
    ]
    
    # Track results
    results = {}
    
    # Run each model sequentially
    for model_name in models:
        logger.info(f"=" * 50)
        logger.info(f"Starting training for model: {model_name}")
        
        # Run the model and track result
        success = run_model(model_name)
        results[model_name] = success
        
        # Clean up between models
        cleanup()
        
        logger.info(f"Completed training for model: {model_name}")
        logger.info(f"=" * 50)
        logger.info("\n") # Add some space between model outputs
    
    # Create summary
    create_summary(results)
    
    logger.info("All training completed!")

if __name__ == "__main__":
    # Set up logger
    logger = setup_logger()
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception in main process: {str(e)}")
    finally:
        logger.info("Run all models script completed.") 