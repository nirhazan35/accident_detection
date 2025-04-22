#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_modified_files():
    """Restore original files and remove unnecessary ones"""
    logger.info("Starting cleanup process...")
    
    # List of modified files to restore (if the original files exist)
    original_files = [
        "src/data_processor.py",
        "src/model.py"
    ]
    
    # List of temporary files to remove
    temp_files = [
        "video_validation.log",
        "test_codec.mp4"
    ]
    
    # List of unnecessary files/folders that can be removed
    unnecessary_files = [
    ]
    
    # Restore original files
    backups_found = False
    for file_path in original_files:
        backup_path = f"{file_path}.bak"
        if os.path.exists(backup_path) and os.path.exists(file_path):
            try:
                logger.info(f"Restoring original file: {file_path}")
                shutil.copy2(backup_path, file_path)
                os.remove(backup_path)
                backups_found = True
            except Exception as e:
                logger.error(f"Error restoring {file_path}: {str(e)}")
    
    if not backups_found:
        logger.info("No backup files found. No files were restored.")
    
    # Remove temporary files
    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                logger.info(f"Removing temporary file: {file_path}")
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing {file_path}: {str(e)}")
    
    # Remove unnecessary files/folders
    for path in unnecessary_files:
        if os.path.exists(path):
            try:
                logger.info(f"Removing unnecessary path: {path}")
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as e:
                logger.error(f"Error removing {path}: {str(e)}")
    
    logger.info("Cleanup completed")

def cleanup_pycache():
    """Remove __pycache__ directories"""
    logger.info("Removing __pycache__ directories...")
    
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_dir = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Removed: {cache_dir}")
                except Exception as e:
                    logger.error(f"Failed to remove {cache_dir}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Clean up temporary and unnecessary files")
    parser.add_argument("--pycache", action="store_true", help="Remove __pycache__ directories")
    args = parser.parse_args()
    
    # Clean up modified files
    cleanup_modified_files()
    
    # Remove __pycache__ if requested
    if args.pycache:
        cleanup_pycache()
    
    logger.info("All cleanup tasks completed")

if __name__ == "__main__":
    main() 