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
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                "cache", "features")
        # 2GB limit for features cache (about 20 songs)
        self.cache_manager = CacheManager(cache_dir, max_files=20, max_size_gb=2.0, max_age_days=90)
        
        # Log cache stats
        stats = self.cache_manager.get_stats()
        logging.info(f"Features cache stats: {stats['file_count']}/{stats['max_files']} files, "
                    f"{stats['total_size_mb']:.1f}/{stats['max_size_gb']*1024:.1f}MB")
        
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
            # Compute hash of audio file
            file_hash = self._compute_file_hash(audio_path)
            
            # Check cache
            cache_path = self.cache_manager.get_from_cache(file_hash)
            if cache_path:
                # Load features from cache
                features = {}
                with np.load(cache_path) as data:
                    for key in data.files:
                        features[key] = data[key]
                return features
                
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
            # Compute hash of audio file
            file_hash = self._compute_file_hash(audio_path)
            
            # Create cache file name
            cache_file = f"{file_hash}.npz"
            
            # Save features to temporary file
            temp_path = os.path.join(self.cache_manager.cache_dir, f"temp_{cache_file}")
            np.savez_compressed(temp_path, **features)
            
            # Add to cache (this will handle cleanup if needed)
            self.cache_manager.add_to_cache(
                file_hash=file_hash,
                original_file=os.path.basename(audio_path),
                cache_file=cache_file,
                metadata={'features': list(features.keys())}
            )
            
            # Move temporary file to final location
            final_path = os.path.join(self.cache_manager.cache_dir, cache_file)
            os.replace(temp_path, final_path)
            
            logging.info(f"Cached features for: {audio_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error caching features: {e}")
            logging.exception("Full traceback:")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
