"""
Handle microphone input processing
"""

import logging
from src.visualization.audio_meter_window import AudioMeterWindow

class MicrophoneProcessor:
    def __init__(self, chunk_callback=None):
        """Initialize microphone processor
        
        Args:
            chunk_callback: Callback function to process audio chunks
        """
        self.chunk_callback = chunk_callback
        self.audio_meter = None
        
    def start_input(self):
        """Setup and start microphone input"""
        try:
            # Create and setup audio meter
            self.audio_meter = AudioMeterWindow()
            if self.chunk_callback:
                self.audio_meter.add_callback(self.chunk_callback)
            
            # Show audio meter window
            self.audio_meter.show()
            logging.info("Microphone input started")
            
        except Exception as e:
            logging.error(f"Error starting microphone input: {e}")
            logging.exception("Full traceback:")
            raise
            
    def stop_input(self):
        """Stop microphone input"""
        if self.audio_meter:
            self.audio_meter.close()
            self.audio_meter = None
            logging.info("Microphone input stopped")
