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
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class AudioFileData:
    waveform: np.ndarray
    sample_rate: int
    duration: float
    num_channels: int
    num_frames: int

class AudioFileProcessor:
    def __init__(self):
        """Initialize audio file processor"""
        self.playback_timer = None
        self.audio_data = None
        self.current_position = 0
        
    def get_audio_info(self, file_path: str) -> tuple[int, float]:
        """Get audio file info without loading the data
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (sample_rate, duration)
        """
        info = torchaudio.info(file_path)
        return info.sample_rate, info.num_frames / info.sample_rate
        
    def load_audio(self, file_path: str, log_info: bool = True) -> AudioFileData:
        """Load and process entire audio file at once
        
        Args:
            file_path: Path to audio file
            log_info: Whether to log info about loading the file
            
        Returns:
            AudioFileData object containing all audio information
            
        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If audio processing fails
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input audio file not found: {file_path}")
                
            if log_info:
                logging.info(f"Loading audio file: {file_path}")
            
            # Get audio info first
            info = torchaudio.info(file_path)
            
            # Load audio file
            waveform, sample_rate = torchaudio.load(file_path)
            waveform = waveform.numpy()
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = np.mean(waveform, axis=0)
            else:
                waveform = waveform[0]
                
            # Convert to float32 if needed
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
                
            # Create audio data object
            audio_data = AudioFileData(
                waveform=waveform,
                sample_rate=sample_rate,
                duration=info.num_frames / info.sample_rate,
                num_channels=info.num_channels,
                num_frames=info.num_frames
            )
            
            # Store for playback if needed
            self.audio_data = audio_data
            
            return audio_data
            
        except Exception as e:
            logging.error(f"Error loading audio file: {e}")
            logging.exception("Full traceback:")
            raise
            
    def start_playback(self, start_time: float = 0.0):
        """Start playing audio from specified time
        
        Args:
            start_time: Time in seconds to start playback from
        """
        try:
            if self.audio_data is None:
                logging.error("No audio data loaded")
                return
                
            # Stop any existing playback
            self.stop_playback()
            
            # Calculate start frame
            start_frame = int(start_time * self.audio_data.sample_rate)
            if start_frame >= len(self.audio_data.waveform):
                logging.error(f"Start time {start_time}s is beyond audio duration")
                return
                
            # Start playback from specified position
            self.current_position = start_frame
            
            # Create timer for UI updates
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self._update_playback_position)
            self.playback_timer.start(100)  # Update every 100ms
            
            # Start audio stream
            try:
                sd.play(
                    self.audio_data.waveform[start_frame:],
                    samplerate=self.audio_data.sample_rate
                )
            except sd.PortAudioError as e:
                QMessageBox.critical(None, "Audio Error",
                                  f"Error playing audio: {str(e)}")
                self.stop_playback()
                
        except Exception as e:
            logging.error(f"Error starting playback: {e}")
            logging.exception("Full traceback:")
            self.stop_playback()
            
    def stop_playback(self):
        """Stop audio playback if active"""
        try:
            if self.playback_timer is not None:
                self.playback_timer.stop()
                self.playback_timer = None
            sd.stop()
        except Exception as e:
            logging.error(f"Error stopping playback: {e}")
            logging.exception("Full traceback:")
            
    def get_current_position(self) -> float:
        """Get current playback position in seconds"""
        return self.current_position / self.audio_data.sample_rate if self.audio_data else 0.0
        
    def _update_playback_position(self):
        """Update playback position for UI updates"""
        try:
            if self.audio_data is None:
                return
                
            # Get current playback position
            latency = sd.get_stream().latency
            self.current_position += int(latency * self.audio_data.sample_rate)
            
            # Stop if we've reached the end
            if self.current_position >= len(self.audio_data.waveform):
                self.stop_playback()
                
        except Exception as e:
            logging.error(f"Error updating playback position: {e}")
            logging.exception("Full traceback:")
