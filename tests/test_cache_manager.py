"""
Simple test script to verify cache manager functionality
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.cache_manager import CacheManager

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_cache_manager():
    # Create a test cache directory
    cache_dir = os.path.join(project_root, "cache", "test_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache manager
    cache_manager = CacheManager(cache_dir, max_files=5, max_size_gb=1.0, max_age_days=30)
    
    # Create some test data
    test_data = "Hello, this is test data!"
    
    # Save test data to a temporary file
    temp_file = os.path.join(cache_dir, "temp_test.txt")
    with open(temp_file, 'w') as f:
        f.write(test_data)
    
    # Get file hash (just use a simple string for testing)
    file_hash = "test_hash_123"
    
    # Add file to cache
    logging.info("Adding file to cache...")
    final_path = cache_manager.add_to_cache(
        file_hash=file_hash,
        original_file="test.txt",
        cache_file="test_hash_123.txt",
        metadata={'test': 'data'}
    )
    
    # Move file to final location
    logging.info(f"Moving file to final location: {final_path}")
    os.replace(temp_file, final_path)
    
    # Try to retrieve the file
    logging.info("Retrieving file from cache...")
    retrieved_path = cache_manager.get_from_cache(file_hash)
    
    if retrieved_path and os.path.exists(retrieved_path):
        logging.info("Successfully retrieved file from cache")
        # Load and verify data
        with open(retrieved_path, 'r') as f:
            data = f.read()
            if data == test_data:
                logging.info("Data matches original")
            else:
                logging.error("Data does not match original")
    else:
        logging.error("Failed to retrieve file from cache")
    
    # Get cache stats
    stats = cache_manager.get_stats()
    logging.info(f"Cache stats: {stats}")
    
    # Check cache index
    with open(os.path.join(cache_dir, "cache_index.json"), 'r') as f:
        index = json.load(f)
        logging.info(f"Cache index: {index}")
    
    # Clean up
    logging.info("Cleaning up...")
    cache_manager.remove_from_cache(file_hash)
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, file))
        os.rmdir(cache_dir)

if __name__ == "__main__":
    setup_logging()
    test_cache_manager()
