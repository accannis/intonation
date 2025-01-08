"""
Audio meter window for the main application
"""

from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import pyqtgraph as pg
import numpy as np
import logging
import sounddevice as sd
import time

class AudioMeterWindow(QMainWindow):
    """Audio meter window for displaying audio levels"""
    
    audio_data = pyqtSignal(np.ndarray, float)  # Signal for audio data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audio Input Level")
        self.setMinimumSize(300, 200)
        
        # Setup UI
        self._setup_ui()
        
        # Initialize audio
        self.stream = None
        self._is_active = False
        self.external_callbacks = []
        self.frame_count = 0
        
        # Keep window on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        
        # Connect signal
        self.audio_data.connect(self._process_audio_data)
        
        # Start audio after a short delay
        QTimer.singleShot(100, self.ensure_audio_started)
        
    def _setup_ui(self):
        """Setup the UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add status label
        self.status_label = QLabel("Monitoring Audio Input...")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Create plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setYRange(0, 1)
        self.plot_widget.setXRange(0, 100)
        self.curve = self.plot_widget.plot(pen='b')
        layout.addWidget(self.plot_widget)
        
        # Buffer for plotting
        self.plot_data = np.zeros(100)
        
    def ensure_audio_started(self):
        """Ensure audio input is started"""
        try:
            if not self._is_active:
                self.start_audio()
                self._is_active = True
        except Exception as e:
            logging.error(f"Error starting audio: {e}")
            logging.exception("Full traceback:")
            
    def start_audio(self):
        """Start audio input stream"""
        try:
            if self.stream is not None:
                self.stop_audio()
                
            def audio_callback(indata, frames, time, status):
                if status:
                    logging.warning(f"Audio callback status: {status}")
                try:
                    # Get audio level
                    level = np.abs(indata[:, 0]).mean()
                    # Emit signal to process in main thread
                    self.audio_data.emit(indata[:, 0], level)
                except Exception as e:
                    logging.error(f"Error in audio callback: {e}")
            
            self.stream = sd.InputStream(
                channels=1,
                samplerate=44100,
                blocksize=2048,
                callback=audio_callback
            )
            self.stream.start()
            logging.info("Audio input started")
            self.status_label.setText("Audio Input Active")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
        except Exception as e:
            logging.error(f"Error starting audio: {e}")
            logging.exception("Full traceback:")
            self.status_label.setText("Error Starting Audio")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
            
    def stop_audio(self):
        """Stop audio input"""
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                self._is_active = False
                logging.info("Audio input stopped")
                self.status_label.setText("Audio Input Stopped")
                self.status_label.setStyleSheet("color: #666; font-style: italic;")
        except Exception as e:
            logging.error(f"Error stopping audio: {e}")
            logging.exception("Full traceback:")
            
    def add_callback(self, callback):
        """Add external callback for audio data"""
        if callback not in self.external_callbacks:
            self.external_callbacks.append(callback)
            
    def remove_callback(self, callback):
        """Remove external callback"""
        if callback in self.external_callbacks:
            self.external_callbacks.remove(callback)
            
    def _process_audio_data(self, audio_data, level):
        """Process audio data in main thread"""
        try:
            # Update plot
            self.plot_data[:-1] = self.plot_data[1:]
            self.plot_data[-1] = level
            self.curve.setData(self.plot_data)
            
            # Call external callbacks
            timestamp = time.time()
            for callback in self.external_callbacks:
                try:
                    callback(audio_data, timestamp)
                except Exception as e:
                    logging.error(f"Error in external callback: {e}")
                    
            self.frame_count += 1
            
        except Exception as e:
            logging.error(f"Error processing audio data: {e}")
            logging.exception("Full traceback:")
            
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_audio()
        super().closeEvent(event)
