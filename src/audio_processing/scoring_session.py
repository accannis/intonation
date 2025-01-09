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
        self.reference_file = config['reference_file']
        
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
                self.input_processor.start_input()
            else:
                # For file input, process everything at once
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
            self.reference_features = self.feature_extractor.extract_features(self.reference_file, include_waveform=False)
            if self.reference_features is None:
                raise RuntimeError("Failed to extract features from reference audio")
                
            # Get duration from audio file metadata without loading the entire file
            sample_rate, duration = self.reference_processor.get_audio_info(self.reference_file)
            
            # Initialize visualizer with reference duration
            self.visualizer.initialize(duration)
            
            logging.info(
                f"Reference features extracted: "
                f"melody shape={self.reference_features['melody'].shape}, "
                f"phonetic shape={self.reference_features['phonetic'].shape}, "
                f"duration={duration:.2f}s"
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
                
            logging.info(
                f"Input features extracted: "
                f"melody shape={input_features['melody'].shape}, "
                f"phonetic shape={input_features['phonetic'].shape}, "
                f"waveform points: {len(input_features['waveform'])}"
            )
            
            # Calculate melody scores over time
            melody_scores, melody_times = self.melody_matcher.match(
                self.reference_features['melody'],
                input_features['melody']
            )
            
            logging.info(f"Melody scores calculated: {len(melody_scores)} points")
            
            # Calculate phonetic scores over time
            phonetic_scores, phonetic_times = self.phonetic_matcher.match(
                self.reference_features['phonetic'],
                input_features['phonetic']
            )
            
            logging.info(f"Phonetic scores calculated: {len(phonetic_scores)} points")
            
            # Calculate current scores (using the last values)
            current_melody_score = melody_scores[-1] if len(melody_scores) > 0 else 0.0
            current_phonetic_score = phonetic_scores[-1] if len(phonetic_scores) > 0 else 0.0
            current_total_score = self.score_calculator.calculate(
                melody_score=current_melody_score,
                phonetic_score=current_phonetic_score
            )
            
            logging.info(
                f"Current scores: "
                f"melody={current_melody_score:.1f}, "
                f"phonetic={current_phonetic_score:.1f}, "
                f"total={current_total_score:.1f}"
            )
            
            # Update visualizer with all scores
            self.visualizer.update_scores(
                total_score=current_total_score,
                melody_score=current_melody_score,
                phonetic_score=current_phonetic_score,
                waveform=input_features['waveform'],
                melody_scores=melody_scores,
                melody_times=melody_times,
                phonetic_scores=phonetic_scores,
                phonetic_times=phonetic_times
            )
            
            logging.info(f"Input file processed: duration={input_data.duration:.2f}s")
            
        except Exception as e:
            logging.error(f"Error processing input file: {e}")
            logging.exception("Full traceback:")
            raise
            
    def process_audio_chunk(self, audio_data, timestamp):
        """Process a chunk of audio data"""
        try:
            # Convert to numpy array if needed
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # Ensure audio data is the right shape
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # Set feature extractor sample rate to match audio
            self.feature_extractor.sample_rate = self.input_processor.sample_rate
            
            # Extract features (include waveform for visualization)
            features = self.feature_extractor.extract_features(audio_data, include_waveform=True)
            if features is None:
                return
                
            # Calculate melody score
            melody_scores, melody_times = self.melody_matcher.match(
                self.reference_features['melody'],
                features['melody']
            )
            
            # Calculate phonetic score
            phonetic_scores, phonetic_times = self.phonetic_matcher.match(
                self.reference_features['phonetic'],
                features['phonetic']
            )
            
            # Calculate current scores (using the last values)
            current_melody_score = melody_scores[-1] if len(melody_scores) > 0 else 0.0
            current_phonetic_score = phonetic_scores[-1] if len(phonetic_scores) > 0 else 0.0
            current_total_score = self.score_calculator.calculate(
                melody_score=current_melody_score,
                phonetic_score=current_phonetic_score
            )
            
            # Update visualizer with current chunk
            self.visualizer.update_scores(
                total_score=current_total_score,
                melody_score=current_melody_score,
                phonetic_score=current_phonetic_score,
                waveform=features['waveform'],
                melody_scores=melody_scores,
                melody_times=melody_times,
                phonetic_scores=phonetic_scores,
                phonetic_times=phonetic_times
            )
            
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")
            logging.exception("Full traceback:")
            
    def cleanup(self):
        """Clean up session resources"""
        try:
            if self.input_processor:
                self.input_processor.stop_playback()
            if self.visualizer:
                self.visualizer.close()
        except Exception as e:
            logging.error(f"Error cleaning up session: {e}")
            logging.exception("Full traceback:")