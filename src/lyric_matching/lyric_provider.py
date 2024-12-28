import logging
from typing import Optional, Dict, Tuple
import requests
import json
import os
import hashlib
from pathlib import Path
import sys

# Add src directory to Python path for imports
src_dir = str(Path(__file__).resolve().parents[1])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.cache_manager import CacheManager

class LyricProvider:
    def __init__(self):
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cache", "lyrics")
        # 10MB limit for lyrics cache (about 5000 songs at ~2KB per song)
        self.cache_manager = CacheManager(cache_dir, max_files=5000, max_size_gb=0.01, max_age_days=90)
        
        # Log cache stats
        stats = self.cache_manager.get_stats()
        logging.info(f"Lyrics cache stats: {stats['file_count']}/{stats['max_files']} files, "
                    f"{stats['total_size_mb']:.1f}/{stats['max_size_gb']*1024:.1f}MB")
        
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def get_lyrics(self, audio_path: str, song_title: str = None, artist: str = None) -> Optional[str]:
        """
        Get lyrics for a song, first trying cache, then online API, then audio extraction
        
        Args:
            audio_path: Path to audio file (used for caching)
            song_title: Optional title of the song
            artist: Optional artist name
            
        Returns:
            Lyrics as string if found, None otherwise
        """
        # Try cache first using audio file hash
        file_hash = self._compute_file_hash(audio_path)
        cached_lyrics_path = self.cache_manager.get_from_cache(file_hash)
        
        if cached_lyrics_path:
            try:
                with open(cached_lyrics_path, 'r', encoding='utf-8') as f:
                    lyrics = f.read()
                    logging.info(f"Found cached lyrics for: {os.path.basename(audio_path)}")
                    return lyrics
            except Exception as e:
                logging.error(f"Error reading lyrics from cache: {e}")
                
        # Try online API if title is provided
        if song_title:
            lyrics = self._fetch_from_api(song_title, artist)
            if lyrics:
                self.save_manual_lyrics(audio_path, lyrics, song_title, artist)
                return lyrics
                
        logging.warning(f"Could not find lyrics for: {os.path.basename(audio_path)}")
        return None
        
    def save_manual_lyrics(self, audio_path: str, lyrics: str, 
                          song_title: str = None, artist: str = None) -> bool:
        """
        Save manually provided lyrics to cache
        
        Args:
            audio_path: Path to audio file
            lyrics: Lyrics text to save
            song_title: Optional song title
            artist: Optional artist name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_hash = self._compute_file_hash(audio_path)
            lyrics_filename = f"{file_hash}.txt"
            lyrics_path = os.path.join(self.cache_manager.cache_dir, lyrics_filename)
            
            # Save lyrics content
            with open(lyrics_path, 'w', encoding='utf-8') as f:
                f.write(lyrics)
                
            # Add to cache
            self.cache_manager.add_to_cache(
                file_hash=file_hash,
                original_file=os.path.basename(audio_path),
                cache_file=lyrics_filename,
                metadata={
                    'song_title': song_title,
                    'artist': artist
                }
            )
            logging.info(f"Cached lyrics for: {os.path.basename(audio_path)}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving manual lyrics: {e}")
            return False
            
    def _fetch_from_api(self, song_title: str, artist: str = None) -> Optional[str]:
        """
        Fetch lyrics from an online API
        TODO: Implement with actual lyrics API (e.g., Genius, Musixmatch)
        """
        # This is a placeholder. We'll need to implement with an actual lyrics API
        return None
        
    def extract_from_audio(self, audio_path: str) -> Optional[str]:
        """
        Extract lyrics from audio using speech recognition
        TODO: Implement with a speech-to-text model
        """
        # This is a placeholder. We'll need to implement with a speech recognition model
        return None
