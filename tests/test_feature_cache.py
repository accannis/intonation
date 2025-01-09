"""
Test script to verify feature cache functionality
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.feature_extraction.feature_cache import FeatureCache

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_feature_cache():
    # Create test audio file
    test_dir = os.path.join(project_root, "tests", "test_data")
    os.makedirs(test_dir, exist_ok=True)
    
    test_audio = os.path.join(test_dir, "test_audio.raw")
    with open(test_audio, 'wb') as f:
        # Create 1 second of random audio data
        samples = np.random.uniform(-1, 1, 44100).astype(np.float32)
        samples.tofile(f)
    
    # Create test features
    test_features = {
        'melody': np.random.rand(100).astype(np.float32),
        'phonetic': np.random.rand(20, 50).astype(np.float32)
    }
    
    # Initialize feature cache
    feature_cache = FeatureCache()
    
    # Try to cache features
    logging.info("Caching features...")
    success = feature_cache.cache_features(test_audio, test_features)
    if not success:
        logging.error("Failed to cache features")
        return
    
    # Try to retrieve features
    logging.info("Retrieving features...")
    cached_features = feature_cache.get_features(test_audio)
    
    if cached_features is None:
        logging.error("Failed to retrieve features from cache")
        return
    
    # Verify features match
    for key in test_features:
        if not np.allclose(test_features[key], cached_features[key]):
            logging.error(f"Cached {key} features do not match original")
            return
    
    logging.info("Features successfully cached and retrieved")
    
    # Clean up
    logging.info("Cleaning up...")
    os.remove(test_audio)
    os.rmdir(test_dir)

if __name__ == "__main__":
    setup_logging()
    test_feature_cache()
