"""
Command-line interface for audio scoring
"""

import argparse
import logging
import json
from pathlib import Path
import numpy as np
import sys
import yaml

from src.audio_processing.scoring_session import ScoringSession
from src.audio_processing.file_processor import AudioFileProcessor
from src.feature_extraction.audio_features import AudioFeatureExtractor
from src.scoring.score_calculator import ScoreCalculator
from src.lyric_matching.phonetic_matcher import PhoneticMatcher
from src.lyric_matching.melody_matcher import MelodyMatcher

def setup_logging():
    """Configure logging for CLI output"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s:%(message)s'
    )

def score_audio(reference_file: str, input_file: str, *, score_melody: bool = True, score_lyrics: bool = True) -> dict:
    """Score input audio against reference
    
    Args:
        reference_file: Path to reference audio file
        input_file: Path to input audio file to score
        score_melody: Whether to compute melody score (default: True)
        score_lyrics: Whether to compute lyrics/phonetic score (default: True)
        
    Returns:
        Dictionary containing scores and analysis
    """
    try:
        # Initialize components with default configs
        feature_config = {
            "sample_rate": 44100,
            "hop_length": 512,
            "n_mels": 128,
            "n_mfcc": 20,
            "n_fft": 2048,
            "mel_power": 2.0,
            "lifter": 22,
            "top_db": 80,
            "f0_min": 65.406391325149666,  # C2
            "f0_max": 2093.004522404789,   # C7
            "delta_width": 9,
            "feature_version": 1
        }
        
        melody_config = {
            "min_note_duration": 0.1,  # seconds
            "min_pitch_confidence": 0.8,
            "pitch_tolerance": 0.5,  # semitones
            "timing_tolerance": 0.2,  # seconds
            "min_overlap": 0.5  # fraction of note duration
        }
        
        phonetic_config = {
            "min_pattern_duration": 0.2,  # seconds
            "max_pattern_duration": 2.0,  # seconds
            "score_threshold": 70.0,  # minimum score to consider a match
            "dtw_window_size": 2.0  # seconds
        }
        
        feature_extractor = AudioFeatureExtractor(feature_config)
        melody_matcher = MelodyMatcher(melody_config) if score_melody else None
        phonetic_matcher = PhoneticMatcher(
            window_size=phonetic_config["dtw_window_size"],
            min_pattern_duration=phonetic_config["min_pattern_duration"],
            max_pattern_duration=phonetic_config["max_pattern_duration"],
            score_threshold=phonetic_config["score_threshold"]
        ) if score_lyrics else None
        
        # Extract reference features (will use cache if available)
        logging.info(f"Processing reference file: {reference_file}")
        reference_features = feature_extractor.extract_features(reference_file)
        if reference_features is None:
            raise RuntimeError("Failed to extract features from reference audio")
            
        # Calculate duration from reference features
        _, mfcc_frames = reference_features['phonetic'].shape
        duration = float(mfcc_frames * 0.05)  # 50ms per frame
            
        # Extract patterns for phonetic matching
        if score_lyrics:
            logging.info("Extracting reference patterns")
            phonetic_matcher.extract_patterns(reference_features['phonetic'], duration=duration)
        
        # Process input file
        logging.info(f"Processing input file: {input_file}")
        input_features = feature_extractor.extract_features(input_file)
        if input_features is None:
            raise RuntimeError("Failed to extract features from input audio")
            
        # Match melody
        melody_score = 0.0
        melody_times = []
        if score_melody:
            logging.info("Matching melody")
            melody_score = melody_matcher.match_melody(reference_features['melody'], input_features['melody'])
            melody_times = [(0, duration)]
            melody_scores = [melody_score]
        
        # Match phonetics
        phonetic_score = 0.0
        phonetic_times = []
        if score_lyrics:
            logging.info("Matching phonetics")
            phonetic_scores, phonetic_times = phonetic_matcher.match_patterns(input_features['phonetic'])
            phonetic_score = max(phonetic_scores) if phonetic_scores else 0.0
        
        # Calculate overall score based on enabled components
        if score_melody and score_lyrics:
            total_score = ScoreCalculator().calculate_total_score(
                melody_score=melody_score,
                phonetic_score=phonetic_score
            )
        elif score_melody:
            total_score = melody_score
        elif score_lyrics:
            total_score = phonetic_score
        else:
            total_score = 0.0
        
        # Prepare results
        results = {
            'scores': {
                'total': float(total_score),
            },
            'analysis': {
                'components_used': {
                    'melody': score_melody,
                    'lyrics': score_lyrics
                }
            }
        }
        
        if score_melody:
            results['scores']['melody'] = float(melody_score)
            
        if score_lyrics:
            results['scores']['phonetic'] = float(phonetic_score)
            
        return results
        
    except Exception as e:
        logging.error(f"Error scoring audio: {e}")
        logging.exception("Full traceback:")
        raise

def score_melody(reference_file: str, input_file: str, 
              feature_extractor: AudioFeatureExtractor,
              melody_matcher: MelodyMatcher) -> float:
    """Score melody similarity between reference and input audio
    
    Args:
        reference_file: Path to reference audio file
        input_file: Path to input audio file
        feature_extractor: Feature extractor instance
        melody_matcher: Melody matcher instance
        
    Returns:
        Melody similarity score between 0 and 100
    """
    # Extract reference features
    print(f"Processing reference file: {reference_file}")
    reference_features = feature_extractor.extract_features(reference_file)
    if reference_features is None:
        print(f"Error: Failed to extract features from reference audio")
        return 0.0

    # Process input file
    print(f"Processing input file: {input_file}")
    input_features = feature_extractor.extract_features(input_file)
    if input_features is None:
        print(f"Error: Failed to extract features from input audio")
        return 0.0

    # Match melody
    print("Matching melody")
    melody_score = melody_matcher.match_melody(reference_features['melody'], input_features['melody'])
    return melody_score

def score_phonetic(reference_file, input_file, feature_extractor, phonetic_matcher):
    # Extract reference features (will use cache if available)
    logging.info(f"Processing reference file: {reference_file}")
    reference_features = feature_extractor.extract_features(reference_file)
    if reference_features is None:
        raise RuntimeError("Failed to extract features from reference audio")
        
    # Calculate duration from reference features
    _, mfcc_frames = reference_features['phonetic'].shape
    duration = float(mfcc_frames * 0.05)  # 50ms per frame
            
    # Extract patterns for phonetic matching
    logging.info("Extracting reference patterns")
    phonetic_matcher.extract_patterns(reference_features['phonetic'], duration=duration)
    
    # Process input file
    logging.info(f"Processing input file: {input_file}")
    input_features = feature_extractor.extract_features(input_file)
    if input_features is None:
        raise RuntimeError("Failed to extract features from input audio")
        
    # Match phonetics
    logging.info("Matching phonetics")
    phonetic_scores, phonetic_times = phonetic_matcher.match(
        reference_features['phonetic'],
        input_features['phonetic']
    )
    
    # Calculate overall score
    phonetic_score = max(phonetic_scores) if phonetic_scores else 0.0
    
    return phonetic_score

def parse_args():
    parser = argparse.ArgumentParser(description='Score an audio file against a reference')
    parser.add_argument('--reference', required=True, help='Reference audio file')
    parser.add_argument('--input', required=True, help='Input audio file to score')
    parser.add_argument('--output', help='Output file for JSON results')
    parser.add_argument('--melody', action='store_true', default=True, help='Enable melody scoring')
    parser.add_argument('--lyrics', action='store_true', default=True, help='Enable lyrics scoring')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info(f"Starting CLI scoring")
    logging.info(f"Reference file: {args.reference}")
    logging.info(f"Input file: {args.input}")
    
    try:
        # Validate files exist
        reference_path = Path(args.reference)
        input_path = Path(args.input)
        
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference file not found: {args.reference}")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
            
        # Create default config
        config = {
            "feature_extraction": {
                "sample_rate": 44100,
                "hop_length": 512,
                "n_mels": 128,
                "n_mfcc": 20,
                "n_fft": 2048,
                "mel_power": 2.0,
                "lifter": 22,
                "top_db": 80,
                "f0_min": 65.406391325149666,  # C2
                "f0_max": 2093.004522404789,   # C7
                "delta_width": 9,
                "feature_version": 1
            },
            "melody_matching": {
                "min_note_duration": 0.1,  # seconds
                "min_pitch_confidence": 0.8,
                "pitch_tolerance": 0.5,  # semitones
                "timing_tolerance": 0.2,  # seconds
                "min_overlap": 0.5  # fraction of note duration
            },
            "phonetic_matching": {
                "min_pattern_duration": 0.2,  # seconds
                "max_pattern_duration": 2.0,  # seconds
                "score_threshold": 70.0,  # minimum score to consider a match
                "dtw_window_size": 2.0  # seconds
            }
        }
        
        # Initialize components
        feature_extractor = AudioFeatureExtractor(config["feature_extraction"])
        melody_score = 0.0
        phonetic_score = 0.0
        
        if args.melody:
            from src.lyric_matching.melody_matcher import MelodyMatcher
            melody_matcher = MelodyMatcher(config["melody_matching"])
            melody_score = score_melody(args.reference, args.input, feature_extractor, melody_matcher)
            print(f"Melody score: {melody_score:.1f}")
            
        if args.lyrics:
            phonetic_matcher = PhoneticMatcher(config["phonetic_matching"])
            phonetic_score = score_phonetic(args.reference, args.input, feature_extractor, phonetic_matcher)
            print(f"Phonetic score: {phonetic_score:.1f}")

        # Return average of enabled scores
        enabled_scores = []
        if args.melody:
            enabled_scores.append(melody_score)
        if args.lyrics:
            enabled_scores.append(phonetic_score)
        final_score = sum(enabled_scores) / len(enabled_scores) if enabled_scores else 0.0
        print(f"Final score: {final_score:.1f}")
        
        # Prepare results
        results = {
            "scores": {
                "total": final_score,
                "melody": melody_score,
                "phonetic": phonetic_score
            },
            "analysis": {
                "melody_detections": [],
                "phonetic_detections": []
            },
            "stats": {
                "reference_duration": 0.0,
                "input_duration": 0.0,
                "n_melody_detections": 0,
                "n_phonetic_detections": 0
            }
        }
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            # Pretty print to console
            print(json.dumps(results, indent=2))
            
        logging.info("Scoring complete")
        return json.dumps(results, indent=2)
            
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.exception("Full traceback:")
        raise SystemExit(1)

if __name__ == '__main__':
    main()
