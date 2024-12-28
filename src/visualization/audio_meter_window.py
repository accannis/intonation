"""
Audio meter window for the main application
"""

from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import Qt
from pyaudiosource import AudioSource, AudioMeter, DeviceManager, AudioTestWindow
import logging

class AudioMeterWindow(AudioTestWindow):
    """Audio meter window that inherits from AudioTestWindow"""
    def __init__(self, parent=None):
        super().__init__()
        self.setParent(parent)
        if parent:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.Tool)
        self.setWindowTitle("Audio Input Level")
        self.external_callbacks = []
        
    def add_callback(self, callback):
        """Add an external callback to be called with audio data"""
        self.external_callbacks.append(callback)
        
    def process_audio(self, indata, level, *args):
        """Process audio data and forward to external callbacks"""
        # Process audio for meters
        super().process_audio(indata, level, *args)
        
        # Forward to external callbacks
        for callback in self.external_callbacks:
            try:
                callback(indata, level)
            except Exception as e:
                logging.error(f"Error in external audio callback: {e}", exc_info=True)
                
    def get_current_device(self):
        """Get the currently selected device index"""
        return self.device_selector.currentData()
        
    def get_current_gain(self):
        """Get the current gain setting"""
        return self.gain_slider.value() / 100.0
