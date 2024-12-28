import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class CacheManager:
    def __init__(self, 
                 cache_dir: str,
                 max_files: int = 20,  # Maximum number of files in cache
                 max_size_gb: float = 2.0,  # Maximum cache size in GB
                 max_age_days: int = 30):  # Maximum age of cache files in days
        
        self.cache_dir = cache_dir
        self.max_files = max_files
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)  # Convert GB to bytes
        self.max_age = timedelta(days=max_age_days)
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache index file
        self.index_path = os.path.join(cache_dir, "cache_index.json")
        self.index = self._load_index()
        
        # Clean cache on startup
        self.cleanup()
        
    def _load_index(self) -> dict:
        """Load the cache index from disk"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading cache index: {e}")
        return {}
        
    def _save_index(self):
        """Save the cache index to disk"""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving cache index: {e}")
            
    def _get_cache_stats(self) -> Tuple[int, int]:
        """Get current cache size in bytes and number of files"""
        total_size = 0
        file_count = 0
        
        for entry in os.scandir(self.cache_dir):
            if entry.is_file() and entry.name != "cache_index.json":
                total_size += entry.stat().st_size
                file_count += 1
                
        return total_size, file_count
        
    def _get_lru_files(self) -> List[Tuple[str, float]]:
        """Get list of files sorted by last access time (oldest first)"""
        files = []
        for file_hash, info in self.index.items():
            file_path = os.path.join(self.cache_dir, info['cache_file'])
            if os.path.exists(file_path):
                last_access = float(info.get('last_accessed', 0))
                files.append((file_hash, last_access))
        return sorted(files, key=lambda x: x[1])
        
    def cleanup(self):
        """Clean up old and excess cache files"""
        current_size, current_files = self._get_cache_stats()
        now = datetime.now().timestamp()
        files_to_remove = []
        
        # First, remove expired files
        for file_hash, info in list(self.index.items()):
            # Handle legacy cache files
            if 'cache_file' not in info:
                # Try to find the file using old naming conventions
                if 'vocals_file' in info:
                    info['cache_file'] = info['vocals_file']
                elif 'lyrics_file' in info:
                    info['cache_file'] = info['lyrics_file']
                else:
                    files_to_remove.append(file_hash)
                    continue
            
            file_path = os.path.join(self.cache_dir, info['cache_file'])
            if not os.path.exists(file_path):
                del self.index[file_hash]
                continue
                
            # Handle legacy timestamps
            if 'last_accessed' not in info:
                if 'created_at' in info:
                    try:
                        # Convert string timestamp to float if needed
                        created_at = float(info['created_at'])
                        info['last_accessed'] = created_at
                    except:
                        info['last_accessed'] = os.path.getmtime(file_path)
                else:
                    info['last_accessed'] = os.path.getmtime(file_path)
                
            last_access = float(info['last_accessed'])
            if now - last_access > self.max_age.total_seconds():
                files_to_remove.append(file_hash)
                
        # Then, remove files if we're over the limit
        if current_files > self.max_files or current_size > self.max_size_bytes:
            lru_files = self._get_lru_files()
            while (current_files > self.max_files or 
                   current_size > self.max_size_bytes) and lru_files:
                file_hash, _ = lru_files.pop(0)
                if file_hash not in files_to_remove:
                    files_to_remove.append(file_hash)
                    info = self.index[file_hash]
                    file_path = os.path.join(self.cache_dir, info['cache_file'])
                    if os.path.exists(file_path):
                        current_size -= os.path.getsize(file_path)
                        current_files -= 1
                        
        # Remove files and update index
        for file_hash in files_to_remove:
            self.remove_from_cache(file_hash)
            
        self._save_index()
        
    def get_from_cache(self, file_hash: str) -> Optional[str]:
        """Get path to cached file if it exists"""
        info = self.index.get(file_hash)
        if info:
            cache_path = os.path.join(self.cache_dir, info['cache_file'])
            if os.path.exists(cache_path):
                # Update last access time
                info['last_accessed'] = datetime.now().timestamp()
                self._save_index()
                return cache_path
        return None
        
    def add_to_cache(self, file_hash: str, original_file: str, 
                     cache_file: str, metadata: dict = None) -> str:
        """
        Add a file to the cache
        
        Args:
            file_hash: Hash of the original file
            original_file: Name of the original file
            cache_file: Name of the file in cache
            metadata: Additional metadata to store
            
        Returns:
            Path to the cached file
        """
        # Clean up if needed
        self.cleanup()
        
        # Add to index
        self.index[file_hash] = {
            'cache_file': cache_file,
            'original_file': original_file,
            'created_at': datetime.now().timestamp(),
            'last_accessed': datetime.now().timestamp()
        }
        if metadata:
            self.index[file_hash].update(metadata)
            
        self._save_index()
        return os.path.join(self.cache_dir, cache_file)
        
    def remove_from_cache(self, file_hash: str):
        """Remove a file from the cache"""
        if file_hash in self.index:
            info = self.index[file_hash]
            file_path = os.path.join(self.cache_dir, info['cache_file'])
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logging.error(f"Error removing cache file: {e}")
            del self.index[file_hash]
            
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_size, file_count = self._get_cache_stats()
        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'max_files': self.max_files,
            'max_size_bytes': self.max_size_bytes,
            'max_size_gb': self.max_size_bytes / (1024 * 1024 * 1024),
            'size_usage_percent': (total_size / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0,
            'file_usage_percent': (file_count / self.max_files) * 100 if self.max_files > 0 else 0
        }
