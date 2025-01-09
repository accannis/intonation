"""
Separate vocals from mixed audio using Demucs
"""

import os
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import numpy as np
from pathlib import Path
import logging
import hashlib
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_dir = str(Path(__file__).resolve().parents[1])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.cache_manager import CacheManager
from src.audio_processing.file_processor import AudioFileProcessor

class VocalSeparator:
    def __init__(self, device='cpu'):
        self.device = device
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cache", "vocals")
        # 2GB limit for vocals cache (about 20 songs at ~100MB per song)
        self.cache_manager = CacheManager(cache_dir, max_files=20, max_size_gb=2.0, max_age_days=30)
        
        logging.info(f"Initializing VocalSeparator with device: {device}")
        logging.info("Loading Demucs model...")
        self.model = get_model('htdemucs')
        self.model.to(device)
        logging.info("Model loaded successfully")
        
        # Create file processor
        self.file_processor = AudioFileProcessor()
        
        # Log cache stats
        stats = self.cache_manager.get_stats()
        logging.info(f"Vocals cache stats: {stats['file_count']}/{stats['max_files']} files, "
                    f"{stats['total_size_mb']:.1f}/{stats['max_size_gb']*1024:.1f}MB")
        
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def separate_vocals(self, audio_path: str, output_path: str) -> str:
        """
        Separate vocals from the mixed audio
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save separated vocals
            
        Returns:
            Path to separated vocals file
        """
        # Load audio using file processor
        audio_data = self.file_processor.load_audio(audio_path, log_info=False)
        wav = torch.from_numpy(audio_data.waveform).to(self.device)
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)  # Add channel dimension if mono
        sr = audio_data.sample_rate
        logging.info(f"Audio loaded successfully. Shape: {wav.shape}, Sample rate: {sr}")
        
        # Apply separation model
        logging.info("Starting vocal separation...")
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        
        with torch.no_grad():
            logging.info("Applying Demucs model...")
            sources = apply_model(self.model, wav[None], device=self.device)[0]
            sources = sources * ref.std() + ref.mean()
            logging.info("Model application complete")
        
        # Get vocals track (index 3 in htdemucs model)
        vocals = sources[3]  # Remove extra dimension
        
        # Save vocals
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, vocals[None], sr)
        logging.info(f"Saved vocals to: {output_path}")
        
        return output_path
        
    def process_song(self, audio_path: str) -> str:
        """
        Process a song to extract vocals
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to separated vocals file
        """
        # Check cache first
        file_hash = self._compute_file_hash(audio_path)
        cached_vocals = self.cache_manager.get_from_cache(file_hash)
        
        if cached_vocals:
            logging.info(f"Using cached vocals for {os.path.basename(audio_path)}")
            return cached_vocals
            
        # Create output path in cache directory
        output_path = os.path.join(self.cache_manager.cache_dir, f"{file_hash}.wav")
        
        # Process audio and save to cache
        output_path = self.separate_vocals(audio_path, output_path)
        
        # Add to cache
        self.cache_manager.add_to_cache(
            file_hash=file_hash,
            original_file=os.path.basename(audio_path),
            cache_file=os.path.basename(output_path)
        )
        
        return output_path
