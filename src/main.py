import os
import sounddevice as sd
import numpy as np
from .feature_extraction.audio_features import AudioFeatureExtractor
from .melody_matching.dtw_matcher import MelodyMatcher
from .lyric_matching.phonetic_matcher import PhoneticMatcher
from .lyric_matching.lyric_provider import LyricProvider
from .scoring.score_calculator import ScoreCalculator
from .preprocessing.vocal_separator import VocalSeparator
import logging
import queue
import threading
import time
from .visualization.score_visualizer import ScoreVisualizer
from .visualization.performance_analyzer import PerformanceAnalyzer
from .visualization.session_player import SessionPlayer
from .visualization.audio_meter_window import AudioMeterWindow
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer
import sys
from datetime import datetime
from pathlib import Path
import json
import h5py
import torchaudio

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# The healing incantation lyrics
HEALING_INCANTATION = """Flower, gleam and glow
Let your powers shine
Make the clock reverse
Bring back what once was mine
Heal what has been hurt
Change the fates' design
Save what has been lost
Bring back what once was mine
What once was mine"""

class SingingScorer:
    def __init__(self, reference_audio_path: str = None, reference_lyrics: str = None):
        """Initialize the singing application"""
        self.reference_audio_path = reference_audio_path
        self.reference_mode = reference_audio_path is not None
        
        logging.info("Initializing Qt application...")
        # Initialize Qt application
        self.app = QApplication(sys.argv)
        
        # Create windows
        logging.info("Creating visualizer...")
        self.visualizer = ScoreVisualizer()
        self.visualizer.audio_source_changed.connect(self._on_audio_source_changed)
        logging.info("Creating performance analyzer...")
        self.performance_analyzer = PerformanceAnalyzer()
        self.session_player = None  # Initialize later on demand
        
        # Create audio meter window (hidden by default)
        logging.info("Creating audio meter window...")
        self.audio_meter = AudioMeterWindow()
        self.audio_meter.add_callback(self.process_audio_chunk)
        
        # Initialize audio processing
        logging.info("Initializing audio processing...")
        self.audio_queue = queue.Queue()
        self.should_stop = False
        self.last_score_time = time.time()
        
        # Initialize components
        logging.info("Initializing audio feature extractor...")
        self.feature_extractor = AudioFeatureExtractor()
        logging.info("Initializing melody matcher...")
        self.melody_matcher = MelodyMatcher()
        logging.info("Initializing phonetic matcher...")
        self.phonetic_matcher = PhoneticMatcher()
        logging.info("Initializing score calculator...")
        self.score_calculator = ScoreCalculator()
        logging.info("Initializing vocal separator...")
        self.vocal_separator = VocalSeparator()
        logging.info("Initializing lyric provider...")
        self.lyric_provider = LyricProvider()
        
        # Only process reference audio if provided
        if self.reference_mode:
            if not os.path.exists(reference_audio_path):
                logging.error(f"Reference audio file not found: {reference_audio_path}")
                QMessageBox.critical(
                    None,
                    "Error",
                    f"Reference audio file not found: {reference_audio_path}\n\nRunning in test mode without reference audio.",
                    QMessageBox.StandardButton.Ok
                )
                self.reference_mode = False
            else:
                logging.info("Processing reference audio...")
                self.process_reference_audio()
        
        # Configure audio parameters
        self.configure_audio()
        
        # Show the window
        self.visualizer.show()
        
    def setup_menu_actions(self):
        """Setup menu actions for the visualizer"""
        # Create File menu
        file_menu = self.visualizer.menuBar().addMenu("File")
        
        # Add "Open Session Player" action
        open_player_action = file_menu.addAction("Open Session Player")
        open_player_action.triggered.connect(self.open_session_player)
        
        # Add "Save Session" action
        save_session_action = file_menu.addAction("Save Session")
        save_session_action.triggered.connect(self.save_session)
        
    def open_session_player(self):
        """Open the session player window"""
        if self.session_player is None:
            self.session_player = SessionPlayer()
        self.session_player.show()
        self.session_player.refresh_sessions()
        
    def save_session(self):
        """Save the current session data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path("session_history") / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save performance data
            with h5py.File(session_dir / "performance_data.h5", 'w') as f:
                # Save scores
                scores_data = np.array([{
                    'total_score': s['total_score'],
                    'melody_score': s['melody_score'],
                    'lyric_score': s['lyric_score']
                } for s in self.score_calculator.score_history])
                
                if len(scores_data) > 0:
                    for key in scores_data[0].dtype.names:
                        f.create_dataset(f'scores/{key}', data=scores_data[key])
                        
                # Save detailed metrics
                metrics_data = self.performance_analyzer.metrics
                for metric, values in metrics_data.items():
                    if values:  # Only save non-empty metrics
                        f.create_dataset(f'metrics/{metric}', data=values)
                        
            # Save performance plots
            metrics_fig, correlation_fig = self.performance_analyzer.plot_performance_trends()
            metrics_fig.savefig(session_dir / "performance_trends.png")
            correlation_fig.savefig(session_dir / "metrics_correlation.png")
            
            # Save session summary
            summary, insights = self.performance_analyzer.generate_performance_report()
            with open(session_dir / "session_summary.json", 'w') as f:
                json.dump({
                    'summary': summary,
                    'insights': insights,
                    'timestamp': timestamp,
                    'reference_audio': str(self.reference_vocals_path) if hasattr(self, 'reference_vocals_path') else None,
                    'settings': {
                        'sample_rate': self.sample_rate,
                        'frame_duration': self.frame_duration,
                        'buffer_duration': self.buffer_duration
                    }
                }, f, indent=2)
                
            # Show success message
            QMessageBox.information(
                self.visualizer,
                "Session Saved",
                f"Session data saved to {session_dir}"
            )
            
            # Refresh session player if open
            if self.session_player is not None and self.session_player.isVisible():
                self.session_player.refresh_sessions()
                
        except Exception as e:
            QMessageBox.critical(
                self.visualizer,
                "Error",
                f"Failed to save session: {str(e)}"
            )
            
        logging.info(f"Session data saved to {session_dir}")
        
    def start_scoring(self):
        """Start real-time scoring"""
        try:
            # Show the window and start Qt event loop
            logging.info("Starting Qt event loop...")
            self.app.exec()
            
        except Exception as e:
            logging.error(f"Error starting scoring: {e}", exc_info=True)
            QMessageBox.critical(
                None,
                "Error",
                f"Error starting scoring: {str(e)}\n\nPlease ensure your microphone is properly connected.",
                QMessageBox.StandardButton.Ok
            )
            raise
            
    def process_audio_chunk(self, indata, level):
        """Callback for processing real-time audio chunks"""
        # Get raw buffer stats before any processing
        raw_min = float(np.min(indata))
        raw_max = float(np.max(indata))
        raw_peak = float(np.max(np.abs(indata)))
        raw_rms = float(np.sqrt(np.mean(np.square(indata))))
        
        logging.info(f"Raw audio: min={raw_min:.4f}, max={raw_max:.4f}, peak={raw_peak:.4f}, rms={raw_rms:.4f}")
        
        # Update the visualizer immediately with the raw level
        if hasattr(self, 'visualizer'):
            self.visualizer.update_data({'total': level * 100}, "", None)
        
        # Add new data to queue along with raw level
        self.audio_queue.put((indata.copy(), level))
        
    def audio_processor(self):
        """Process audio data from the queue"""
        logging.info("Starting audio processor thread")
        while not self.should_stop:
            try:
                # Get data from queue with timeout
                data = self.audio_queue.get(timeout=0.1)
                
                # Unpack the tuple if it is one
                if isinstance(data, tuple):
                    indata, raw_level = data
                else:
                    indata = data
                
                # Roll the buffer and add new data
                samples_to_add = len(indata)
                self.audio_buffer = np.roll(self.audio_buffer, -samples_to_add)
                self.audio_buffer[-samples_to_add:] = indata.flatten()
                
                # Process every score_interval seconds
                current_time = time.time()
                if current_time - self.last_score_time >= self.score_interval:
                    logging.debug("Processing audio buffer...")
                    
                    # Normalize the buffer
                    if np.max(np.abs(self.audio_buffer)) > 0:
                        normalized_buffer = self.audio_buffer / np.max(np.abs(self.audio_buffer))
                    else:
                        normalized_buffer = self.audio_buffer
                    
                    # Extract features
                    features = self.feature_extractor.extract_features_from_buffer(
                        normalized_buffer,
                        self.sample_rate
                    )
                    if not features:
                        logging.debug("No features extracted, skipping frame")
                        continue
                    
                    # Update visualization with features
                    self.visualizer.update_data(
                        audio_buffer=normalized_buffer,
                        features=features
                    )
                    
                    self.last_score_time = current_time
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing audio: {e}", exc_info=True)
                continue
                
    def process_reference_audio(self):
        """Process the reference audio"""
        # Always use the healing incantation lyrics
        logging.info("Using healing incantation lyrics...")
        reference_lyrics = HEALING_INCANTATION
        self.lyric_provider.save_manual_lyrics(
            self.reference_audio_path,
            reference_lyrics,
            song_title="Healing Incantation"
        )
        
        # Process reference lyrics
        logging.info("Processing reference lyrics...")
        self.reference_phonemes = self.phonetic_matcher.text_to_phonemes(
            reference_lyrics
        )
        logging.info(f"Reference lyrics: {reference_lyrics}")
        logging.info(f"Expected phonemes: {self.reference_phonemes}")
        
        # First separate vocals from the reference audio
        logging.info("Separating vocals from reference audio...")
        self.reference_vocals_path = self.vocal_separator.process_song(self.reference_audio_path)
        
        # Load and process reference audio
        logging.info("Extracting features from reference vocals...")
        self.reference_features = self.feature_extractor.extract_features(
            self.reference_vocals_path
        )
        logging.info(f"Reference features extracted: {list(self.reference_features.keys())}")
        
    def configure_audio(self):
        """Configure audio parameters"""
        # Real-time processing parameters
        self.sample_rate = 44100
        self.frame_duration = 0.05  # 50ms frames
        self.buffer_duration = 2.0   # 2 second buffer
        self.frame_size = int(self.sample_rate * self.frame_duration)
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        self.audio_buffer = np.zeros(self.buffer_size)
        self.gain = 1.0
        self.current_device = None
        self.stream = None
        
        # Performance tracking
        self.frames_processed = 0
        self.last_score_time = 0
        self.score_interval = 0.5  # Update score every 0.5 seconds
        
        # Audio processing queue and thread
        self.processing_thread = None
        self.should_stop = False
        
    def _on_audio_source_changed(self, source_type: str, file_path: str):
        """Handle audio source changes"""
        if source_type == "input":
            # Switch to audio input mode
            logging.info("Switching to audio input mode")
            self.audio_meter.show()
            self.stop_file_playback()
        else:
            # Switch to file mode
            logging.info(f"Switching to file mode: {file_path}")
            self.audio_meter.hide()
            self.start_file_playback(file_path)
            
    def start_file_playback(self, file_path: str):
        """Start playing and processing audio from file"""
        try:
            # Stop any existing playback
            self.stop_file_playback()
            
            # Load the audio file
            logging.info(f"Loading audio file: {file_path}")
            self.audio_data, self.sample_rate = torchaudio.load(file_path)
            self.audio_data = self.audio_data.numpy()
            
            # Convert to mono if stereo
            if self.audio_data.shape[0] > 1:
                self.audio_data = np.mean(self.audio_data, axis=0)
            else:
                self.audio_data = self.audio_data[0]
                
            # Initialize playback position
            self.current_position = 0
            self.frame_size = int(self.sample_rate * 0.02)  # 20ms chunks
            
            # Start playback timer
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self._process_next_chunk)
            self.playback_timer.start(20)  # 20ms intervals
            
            logging.info("File playback started")
            
        except Exception as e:
            logging.error(f"Error starting file playback: {e}", exc_info=True)
            QMessageBox.critical(
                None,
                "Error",
                f"Error playing audio file: {str(e)}",
                QMessageBox.StandardButton.Ok
            )
            
    def stop_file_playback(self):
        """Stop file playback if active"""
        if hasattr(self, 'playback_timer') and self.playback_timer is not None:
            self.playback_timer.stop()
            self.playback_timer = None
            logging.info("File playback stopped")
            
    def _process_next_chunk(self):
        """Process next chunk of audio from file"""
        if not hasattr(self, 'current_position') or not hasattr(self, 'audio_data'):
            self.stop_file_playback()
            return
            
        # Get next chunk
        start = self.current_position
        end = start + self.frame_size
        
        if end >= len(self.audio_data):
            # Reached end of file
            self.stop_file_playback()
            return
            
        chunk = self.audio_data[start:end]
        
        # Calculate level
        level = np.max(np.abs(chunk))
        
        # Process chunk
        self.process_audio_chunk(chunk, level)
        
        # Update position
        self.current_position = end
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Use the provided reference audio file
    reference_audio = os.path.join(os.path.dirname(os.path.dirname(__file__)), "healingincarnation.wav")
    logging.info(f"Looking for reference audio at: {reference_audio}")
    app = SingingScorer(reference_audio)  # Pass the reference audio path
    app.start_scoring()
