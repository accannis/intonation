"""
Manages a single scoring session with its components and visualization
"""

import os
import logging
import queue
import numpy as np
from typing import Dict, Optional

from src.feature_extraction.audio_features import AudioFeatureExtractor
from src.melody_matching.dtw_matcher import MelodyMatcher
from src.lyric_matching.phonetic_matcher import PhoneticMatcher
from src.scoring.score_calculator import ScoreCalculator
from src.preprocessing.vocal_separator import VocalSeparator
from src.lyric_matching.lyric_provider import LyricProvider
from src.visualization.score_visualizer import ScoreVisualizer
from src.audio_processing.file_processor import AudioFileProcessor
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
        
        # Create audio processors
        self.reference_processor = AudioFileProcessor(chunk_callback=None)  # Reference processor doesn't need callback
        if self.input_source == 'file':
            self.input_processor = AudioFileProcessor(chunk_callback=self.process_audio_chunk)
        else:
            self.input_processor = MicrophoneProcessor(chunk_callback=self.process_audio_chunk)
        
        # Create visualizer
        self.visualizer = ScoreVisualizer()
        
        # Initialize processing queue
        self.audio_queue = queue.Queue()
        
    def _start_session(self):
        """Start the scoring session"""
        try:
            # Validate reference file
            if not os.path.exists(self.reference_file):
                raise FileNotFoundError(f"Reference audio file not found: {self.reference_file}")
                
            # Process reference audio
            self._process_reference_audio()
            
            # Show visualizer
            self.visualizer.show()
            
            # Start audio input
            if self.input_source == 'microphone':
                self.input_processor.start_input()
            else:
                self.input_processor.start_playback(self.input_file)
                
        except Exception as e:
            logging.error(f"Error starting session: {e}")
            logging.exception("Full traceback:")
            raise
            
    def _process_reference_audio(self):
        """Process the reference audio file"""
        try:
            # Load and process reference audio
            waveform, sample_rate, chunk_size, num_chunks = self.reference_processor.load_reference_audio(
                self.reference_file
            )
            
            reference_features = None
            
            # Process each chunk
            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk = waveform[start:end]
                
                # Extract features from chunk
                chunk_features = self.feature_extractor.extract_features(chunk)
                if chunk_features is None:
                    continue
                    
                # Initialize reference features with first chunk
                if reference_features is None:
                    reference_features = {
                        'mfccs': chunk_features['mfccs'],
                        'mel_spec': chunk_features['mel_spec'],
                        'pitch': chunk_features['pitch']
                    }
                else:
                    # Concatenate features
                    reference_features['mfccs'] = np.concatenate([reference_features['mfccs'], chunk_features['mfccs']], axis=1)
                    reference_features['mel_spec'] = np.concatenate([reference_features['mel_spec'], chunk_features['mel_spec']], axis=1)
                    reference_features['pitch'] = np.concatenate([reference_features['pitch'], chunk_features['pitch']])
            
            if reference_features is None:
                raise Exception("Failed to extract features from reference audio")
            
            # Set reference for matchers
            self.melody_matcher.set_reference(reference_features)
            self.phonetic_matcher.set_reference(reference_features)
            
            # Initialize score calculator
            self.score_calculator.set_weights(melody_weight=0.6, phonetic_weight=0.4)
            
        except Exception as e:
            logging.error(f"Error processing reference audio: {e}")
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
            
            # Extract features
            features = self.feature_extractor.extract_features(audio_data)
            if features is None:
                logging.warning("Failed to extract features from audio chunk")
                return
                
            # Calculate scores
            melody_score = self.melody_matcher.calculate_score(features)
            phonetic_score = self.phonetic_matcher.calculate_score(features)
            total_score = self.score_calculator.calculate_total_score(melody_score, phonetic_score)
            
            # Create feature dictionary
            feature_dict = {
                'melody_score': melody_score,
                'phonetic_score': phonetic_score,
                'total_score': total_score,
                'feedback': f"Melody: {melody_score:.1f}, Phonetic: {phonetic_score:.1f}"
            }
            
            # Update visualizer
            self.visualizer.update_display(feature_dict, timestamp, audio_data)
            
            # Store in queue for analysis
            self.audio_queue.put((feature_dict, timestamp))
            
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")
            logging.exception("Full traceback:")
            
    def cleanup(self):
        """Clean up session resources"""
        if hasattr(self, 'visualizer'):
            self.visualizer.close()
        
        if self.input_source == 'microphone':
            self.input_processor.stop_input()
        else:
            self.input_processor.stop_playback()