"""
Manages a single scoring session with its components and visualization
"""

import os
import logging
import queue
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta

from src.feature_extraction.audio_features import AudioFeatureExtractor
from src.melody_matching.dtw_matcher import MelodyMatcher
from src.lyric_matching.phonetic_matcher import PhoneticMatcher
from src.scoring.score_calculator import ScoreCalculator
from src.preprocessing.vocal_separator import VocalSeparator
from src.lyric_matching.lyric_provider import LyricProvider
from src.visualization.score_visualizer import ScoreVisualizer
from src.audio_processing.file_processor import AudioFileProcessor, AudioFileData
from src.audio_processing.mic_processor import MicrophoneProcessor

class ScoringSession:
    def __init__(self, config: Dict):
        """Initialize a new scoring session
        
        Args:
            config: Dictionary containing session configuration:
                   - input_source: 'microphone' or 'file'
                   - input_file: Path to input file (if using file input)
                   - reference_file: Path to reference audio file
        """
        self.config = config
        self.input_source = config['input_source']
        self.input_file = config.get('input_file')
        if self.input_file:
            self.input_file = os.path.abspath(self.input_file)
        self.reference_file = os.path.abspath(config['reference_file'])
        
        # Initialize processing components
        self._init_components()
        
        # Process reference audio and start session
        self._start_session()
        
    def _init_components(self):
        """Initialize all session components"""
        # Create processing components
        self.feature_extractor = AudioFeatureExtractor()
        self.melody_matcher = MelodyMatcher()
        self.phonetic_matcher = PhoneticMatcher()
        self.score_calculator = ScoreCalculator()
        self.vocal_separator = VocalSeparator()
        self.lyric_provider = LyricProvider()
        
        # Create visualizer
        self.visualizer = ScoreVisualizer()
        self.visualizer.show()
        
        # Create audio processors
        self.reference_processor = AudioFileProcessor()
        if self.input_source == 'file':
            self.input_processor = AudioFileProcessor()
        else:
            self.input_processor = MicrophoneProcessor(chunk_callback=self.process_audio_chunk)
        
        # Initialize processing queue for microphone input
        self.audio_queue = queue.Queue()
            
    def _start_session(self):
        """Start the scoring session"""
        try:
            # Process reference audio first
            self._process_reference_audio()
            
            # Handle input based on source
            if self.input_source == 'microphone':
                # For microphone input, initialize visualizer with a default duration
                # that will expand as needed
                self.visualizer.initialize(duration=30.0)  # Start with 30 seconds
                self.input_processor.start_input()
            else:
                # For file input, get input file duration
                sample_rate, input_duration = self.input_processor.get_audio_info(self.input_file)
                self.visualizer.initialize(duration=input_duration)
                
                # Process entire input file
                self._process_input_file()
                # Start playback for user feedback
                self.input_processor.start_playback()
                
        except Exception as e:
            logging.error(f"Error starting session: {e}")
            logging.exception("Full traceback:")
            raise
            
    def _process_reference_audio(self):
        """Process the reference audio file"""
        try:
            # Try to get features from cache first
            logging.debug(f"Processing reference audio file: {self.reference_file}")
            self.reference_features = self.feature_extractor.extract_features(self.reference_file, include_waveform=False)
            if self.reference_features is None:
                raise RuntimeError("Failed to extract features from reference audio")
                
            logging.info(
                f"Reference features extracted: "
                f"path={self.reference_file}, "
                f"melody shape={self.reference_features['melody'].shape}, "
                f"phonetic shape={self.reference_features['phonetic'].shape}"
            )
            
        except Exception as e:
            logging.error(f"Error processing reference audio: {e}")
            logging.exception("Full traceback:")
            raise
            
    def _process_input_file(self):
        """Process input audio file all at once"""
        try:
            # Load and process entire input audio
            input_data = self.input_processor.load_audio(self.input_file)
            
            # Extract features from entire input audio (include waveform for visualization)
            input_features = self.feature_extractor.extract_features(input_data.waveform, include_waveform=True)
            if input_features is None:
                raise RuntimeError("Failed to extract features from input audio")
                
            # Process features
            self.process_features(input_features, input_data.duration)
            
        except Exception as e:
            logging.error(f"Error processing input file: {e}")
            logging.exception("Full traceback:")
            raise
            
    def process_audio_chunk(self, audio_data: AudioFileData, timestamp: float):
        """Process a chunk of audio data"""
        try:
            # Extract features from audio chunk
            chunk_features = self.feature_extractor.extract_features(audio_data.waveform, include_waveform=True)
            if chunk_features is None:
                logging.warning("Failed to extract features from audio chunk")
                return
                
            # Process features
            self.process_features(chunk_features, audio_data.duration, timestamp)
            
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")
            logging.exception("Full traceback:")
            
    def process_features(self, features: Dict[str, np.ndarray], duration: float, timestamp: float = 0):
        """Process extracted features"""
        try:
            # Match melody
            melody_scores, melody_times = self.melody_matcher.match(
                self.reference_features['melody'],
                features['melody']
            )
            
            # Match phonetics
            phonetic_scores, phonetic_times = self.phonetic_matcher.match(
                self.reference_features['phonetic'],
                features['phonetic']
            )
            
            # Calculate current scores (using the last values)
            current_melody_score = melody_scores[-1] if len(melody_scores) > 0 else 0.0
            current_phonetic_score = phonetic_scores[-1] if len(phonetic_scores) > 0 else 0.0
            current_total_score = self.score_calculator.calculate_total_score(
                melody_score=current_melody_score,
                phonetic_score=current_phonetic_score
            )
            
            # Update visualization
            self.visualizer.update_scores(
                waveform=features['waveform'],
                melody_score=current_melody_score,
                phonetic_score=current_phonetic_score,
                total_score=current_total_score,
                duration=duration,
                timestamp=timestamp
            )
            
        except Exception as e:
            logging.error(f"Error processing features: {e}")
            logging.exception("Full traceback:")
            
    def cleanup(self):
        """Clean up session resources"""
        try:
            if self.input_processor:
                self.input_processor.stop()
            if self.visualizer:
                self.visualizer.close()
        except Exception as e:
            logging.error(f"Error cleaning up session: {e}")
            logging.exception("Full traceback:")