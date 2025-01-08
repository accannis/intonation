"""
Handle audio file processing and playback
"""

import os
import logging
import numpy as np
import torchaudio
import sounddevice as sd
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox

class AudioFileProcessor:
    def __init__(self, chunk_callback=None):
        """Initialize audio file processor
        
        Args:
            chunk_callback: Callback function to process audio chunks
        """
        self.chunk_callback = chunk_callback
        self.playback_timer = None
        self.audio_data = None
        self.sample_rate = None
        self.current_position = 0
        
    def start_playback(self, file_path):
        """Start playing and processing audio from file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input audio file not found: {file_path}")
                
            logging.info(f"Loading audio file: {file_path}")
            
            # Load audio file
            self.audio_data, self.sample_rate = torchaudio.load(file_path)
            self.audio_data = self.audio_data.numpy()
            
            # Convert to mono if stereo
            if self.audio_data.shape[0] > 1:
                self.audio_data = np.mean(self.audio_data, axis=0)
            else:
                self.audio_data = self.audio_data[0]
                
            # Convert to float32
            self.audio_data = self.audio_data.astype(np.float32)
                
            # Initialize playback position
            self.current_position = 0
            self.frame_size = int(self.sample_rate * 0.02)  # 20ms chunks
            
            # Start playback timer
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self._process_next_chunk)
            self.playback_timer.start(20)  # 20ms intervals
            
            logging.info("File playback started")
            
        except Exception as e:
            logging.error(f"Error starting file playback: {e}")
            logging.exception("Full traceback:")
            QMessageBox.critical(
                None,
                "Playback Error",
                f"Error playing audio file: {str(e)}",
                QMessageBox.StandardButton.Ok
            )
            
    def stop_playback(self):
        """Stop file playback if active"""
        if self.playback_timer is not None:
            self.playback_timer.stop()
            self.playback_timer = None
            logging.info("File playback stopped")
            
    def _process_next_chunk(self):
        """Process next chunk of audio from file"""
        if self.audio_data is None:
            self.stop_playback()
            return
            
        try:
            # Get next chunk
            start = self.current_position
            end = start + self.frame_size
            
            if end >= len(self.audio_data):
                # Reached end of file
                self.stop_playback()
                return
                
            chunk = self.audio_data[start:end]
            self.current_position = end
            
            # Process chunk
            timestamp = start / self.sample_rate
            if self.chunk_callback:
                self.chunk_callback(chunk, timestamp)
            
            # Play audio through speakers
            sd.play(chunk, self.sample_rate, blocking=False)
            
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")
            logging.exception("Full traceback:")
            self.stop_playback()
            
    def load_reference_audio(self, file_path, chunk_size_ms=20):
        """Load and process reference audio file in chunks
        
        Args:
            file_path: Path to reference audio file
            chunk_size_ms: Size of chunks in milliseconds
            
        Returns:
            Dictionary of accumulated features from all chunks
        """
        try:
            logging.info(f"Processing reference audio: {file_path}")
            
            # Load reference audio
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Convert to numpy array and handle stereo
            waveform = waveform.numpy()
            if waveform.shape[0] > 1:
                waveform = np.mean(waveform, axis=0)
            else:
                waveform = waveform[0]
                
            # Convert to float32
            waveform = waveform.astype(np.float32)
            
            # Process in chunks
            chunk_size = int(sample_rate * chunk_size_ms / 1000)
            num_chunks = len(waveform) // chunk_size
            
            return waveform, sample_rate, chunk_size, num_chunks
            
        except Exception as e:
            logging.error(f"Error loading reference audio: {e}")
            logging.exception("Full traceback:")
            raise
