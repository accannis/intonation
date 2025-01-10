"""
Cache manager for audio features
"""

import os
import logging
import numpy as np
from typing import Dict, Optional
import hashlib
import json
import time

from src.utils.cache_manager import CacheManager

class FeatureCache:
    def __init__(self):
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                "cache", "features")
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize index
        self.index = {}
        
        # 2GB limit for features cache (about 20 songs)
        self.cache_manager = CacheManager(self.cache_dir, max_files=20, max_size_gb=2.0, max_age_days=90)
        
        # Log cache stats
        stats = self.cache_manager.get_stats()
        logging.info(f"Features cache stats: {stats['file_count']}/{stats['max_files']} files, "
                    f"{stats['total_size_mb']:.1f}/{stats['max_size_gb']*1024:.1f}MB")
        logging.info(f"Cache directory: {self.cache_dir}")
        if os.path.exists(self.cache_dir):
            logging.info(f"Cache directory contents: {os.listdir(self.cache_dir)}")
        
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def _compute_cache_key(self, file_path: str, parameters: Dict) -> str:
        """Compute cache key from file hash and parameters
        
        Args:
            file_path: Path to audio file
            parameters: Dictionary of feature extraction parameters
            
        Returns:
            Cache key string
        """
        try:
            # Get file hash
            file_hash = self._compute_file_hash(file_path)
            logging.debug(f"File hash: {file_hash}")
            
            # Sort parameters to ensure consistent order
            param_str = json.dumps(parameters, sort_keys=True)
            logging.debug(f"Parameter string: {param_str}")
            
            # Combine file hash and parameters
            combined = f"{file_hash}_{param_str}"
            
            # Hash the combined string
            cache_key = hashlib.md5(combined.encode()).hexdigest()
            logging.debug(f"Generated cache key: {cache_key}")
            
            return cache_key
            
        except Exception as e:
            logging.error(f"Error computing cache key: {e}")
            logging.exception("Full traceback:")
            # Return a unique key to prevent cache hits
            return f"error_{time.time()}"
        
    def get_features(self, audio_path: str, parameters: Dict) -> Optional[Dict[str, np.ndarray]]:
        """Get cached features for an audio file
        
        Args:
            audio_path: Path to audio file
            parameters: Dictionary of feature extraction parameters
            
        Returns:
            Dictionary of features if found in cache, None otherwise
        """
        try:
            # Log full path
            abs_path = os.path.abspath(audio_path)
            logging.debug(f"Getting features for {abs_path}")
            
            # Compute cache key
            cache_key = self._compute_cache_key(abs_path, parameters)
            logging.debug(f"Cache key: {cache_key}")
            
            # Check if features exist in cache
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.npz")
            if not os.path.exists(cache_file):
                logging.debug(f"No cached features found for {os.path.basename(abs_path)}")
                return None
                
            # Load features from cache
            logging.debug(f"Loading cached features from {cache_file}")
            with np.load(cache_file) as data:
                features = {k: data[k] for k in data.files}
                
            return features
            
        except Exception as e:
            logging.error(f"Error getting cached features: {e}")
            logging.exception("Full traceback:")
            return None
            
    def cache_features(self, audio_path: str, parameters: Dict, features: Dict[str, np.ndarray]) -> bool:
        """Cache features for an audio file
        
        Args:
            audio_path: Path to audio file
            parameters: Dictionary of feature extraction parameters
            features: Dictionary of features to cache
            
        Returns:
            True if features were cached successfully, False otherwise
        """
        try:
            # Get absolute path
            abs_path = os.path.abspath(audio_path)
            
            # Compute cache key
            cache_key = self._compute_cache_key(abs_path, parameters)
            
            # Save features to cache
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.npz")
            np.savez_compressed(cache_file, **features)
            
            # Update cache index
            self.index[cache_key] = {
                'file_path': abs_path,
                'parameters': parameters,
                'timestamp': os.path.getmtime(abs_path),
                'cache_file': cache_file
            }
            
            # Save updated index
            with open(self.index_path, 'w') as f:
                json.dump(self.index, f)
            
            return True
            
        except Exception as e:
            logging.error(f"Error caching features: {e}")
            logging.exception("Full traceback:")
            return False

    def clear_cache(self):
        """Clear all cached features"""
        try:
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.npy'):
                        os.remove(os.path.join(self.cache_dir, file))
            logging.info("Feature cache cleared")
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")
