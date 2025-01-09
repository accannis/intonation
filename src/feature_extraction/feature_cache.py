"""
Cache manager for audio features
"""

import os
import logging
import numpy as np
from typing import Dict, Optional
import hashlib
import json

from src.utils.cache_manager import CacheManager

class FeatureCache:
    def __init__(self):
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                "cache", "features")
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
        
    def get_features(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Get cached features for an audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of features if found in cache, None otherwise
        """
        try:
            # Log full path
            abs_path = os.path.abspath(audio_path)
            logging.info(f"Looking for cached features at path: {abs_path}")
            
            # Compute hash of audio file
            file_hash = self._compute_file_hash(abs_path)
            logging.info(f"Computed hash for {os.path.basename(abs_path)}: {file_hash}")
            
            # Check cache
            cache_path = self.cache_manager.get_from_cache(file_hash)
            logging.info(f"Cache path for {file_hash}: {cache_path}")
            
            if cache_path and os.path.exists(cache_path):
                logging.info(f"Cache hit for {os.path.basename(abs_path)} - loading features from cache")
                try:
                    # Load features from cache
                    with np.load(cache_path) as data:
                        features = {key: data[key] for key in data.files}
                        logging.info(f"Loaded features from cache: {list(features.keys())}")
                        return features
                except Exception as e:
                    logging.error(f"Error loading cached features: {e}")
                    # Remove corrupted cache file
                    self.cache_manager.remove_from_cache(file_hash)
            else:
                logging.info(f"Cache miss for {os.path.basename(abs_path)} - features will be extracted")
                if not cache_path:
                    logging.info("Cache miss reason: No cache path returned")
                elif not os.path.exists(cache_path):
                    logging.info(f"Cache miss reason: Cache file does not exist at {cache_path}")
                    logging.info(f"Cache directory contents: {os.listdir(self.cache_dir)}")
                
        except Exception as e:
            logging.error(f"Error getting features from cache: {e}")
            logging.exception("Full traceback:")
            
        return None
        
    def cache_features(self, audio_path: str, features: Dict[str, np.ndarray]) -> bool:
        """Cache features for an audio file
        
        Args:
            audio_path: Path to audio file
            features: Dictionary of features to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            # Log full path
            abs_path = os.path.abspath(audio_path)
            logging.info(f"Caching features for path: {abs_path}")
            
            # Compute hash of audio file
            file_hash = self._compute_file_hash(abs_path)
            logging.info(f"Caching features for {os.path.basename(abs_path)} with hash {file_hash}")
            
            # Create cache file name
            cache_file = f"{file_hash}.npz"
            temp_path = os.path.join(self.cache_dir, f"temp_{cache_file}")
            logging.info(f"Saving features to temporary file: {temp_path}")
            
            # Ensure cache directory exists
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            logging.info(f"Cache directory created: {os.path.dirname(temp_path)}")
            
            # Save features to temporary file
            try:
                np.savez_compressed(temp_path, **features)
                logging.info(f"Successfully saved features to temporary file")
            except Exception as e:
                logging.error(f"Error saving features to cache: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
            
            # Add to cache (this will handle cleanup if needed)
            logging.info(f"Adding file to cache manager: {cache_file}")
            final_path = self.cache_manager.add_to_cache(
                file_hash=file_hash,
                original_file=os.path.basename(abs_path),
                cache_file=cache_file,
                metadata={'features': list(features.keys())}
            )
            logging.info(f"Cache manager returned final path: {final_path}")
            
            # Ensure final directory exists
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            logging.info(f"Final directory created: {os.path.dirname(final_path)}")
            
            # Move temporary file to final location
            try:
                os.replace(temp_path, final_path)
                logging.info(f"Cached features for {os.path.basename(abs_path)}")
                logging.info(f"Successfully moved cache file to final location: {final_path}")
                return True
            except Exception as e:
                logging.error(f"Error moving cache file to final location: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                self.cache_manager.remove_from_cache(file_hash)
                return False
                
        except Exception as e:
            logging.error(f"Error caching features: {e}")
            logging.exception("Full traceback:")
            return False
