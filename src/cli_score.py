"""
Command-line interface for audio scoring
"""

import argparse
import logging
import json
from pathlib import Path
import numpy as np

from src.audio_processing.scoring_session import ScoringSession
from src.audio_processing.file_processor import AudioFileProcessor
from src.feature_extraction.audio_features import AudioFeatureExtractor
from src.scoring.score_calculator import ScoreCalculator
from src.lyric_matching.phonetic_matcher import PhoneticMatcher
from src.melody_matching.dtw_matcher import MelodyMatcher

def setup_logging():
    """Configure logging for CLI output"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s:%(message)s'
    )

def score_audio(reference_file: str, input_file: str) -> dict:
    """Score input audio against reference
    
    Args:
        reference_file: Path to reference audio file
        input_file: Path to input audio file to score
        
    Returns:
        Dictionary containing scores and analysis
    """
    try:
        # Initialize components
        feature_extractor = AudioFeatureExtractor()
        phonetic_matcher = PhoneticMatcher()
        melody_matcher = MelodyMatcher()
        score_calculator = ScoreCalculator()
        file_processor = AudioFileProcessor()
        
        # Extract reference features (will use cache if available)
        logging.info(f"Processing reference file: {reference_file}")
        reference_features = feature_extractor.extract_features(reference_file)
        if reference_features is None:
            raise RuntimeError("Failed to extract features from reference audio")
            
        # Extract patterns for phonetic matching
        logging.info("Extracting reference patterns")
        _, mfcc_frames = reference_features['phonetic'].shape
        duration = mfcc_frames * 0.05  # 50ms per frame
        
        # Break reference into 2-second chunks
        chunk_duration = 2.0
        pattern_times = []
        for start_time in np.arange(0, duration, chunk_duration):
            end_time = min(start_time + chunk_duration, duration)
            pattern_id = f"chunk_{len(pattern_times)}"
            pattern_times.append((pattern_id, start_time, end_time))
            
        phonetic_matcher.extract_patterns(reference_features['phonetic'], pattern_times)
        
        # Process input file
        logging.info(f"Processing input file: {input_file}")
        input_features = feature_extractor.extract_features(input_file)
        if input_features is None:
            raise RuntimeError("Failed to extract features from input audio")
            
        # Match melody
        logging.info("Matching melody")
        melody_scores, melody_times = melody_matcher.match(
            reference_features['melody'],
            input_features['melody']
        )
        
        # Match phonetics
        logging.info("Matching phonetics")
        phonetic_scores, phonetic_times = phonetic_matcher.match(
            reference_features['phonetic'],
            input_features['phonetic']
        )
        
        # Calculate overall score
        melody_score = max(melody_scores) if melody_scores else 0.0
        phonetic_score = max(phonetic_scores) if phonetic_scores else 0.0
        total_score = score_calculator.calculate_total_score(
            melody_score=melody_score,
            phonetic_score=phonetic_score
        )
        
        # Prepare detailed results
        results = {
            'scores': {
                'total': float(total_score),
                'melody': float(melody_score),
                'phonetic': float(phonetic_score)
            },
            'analysis': {
                'melody_detections': [
                    {'time': float(t), 'score': float(s)} 
                    for t, s in zip(melody_times, melody_scores)
                ],
                'phonetic_detections': [
                    {'time': float(t), 'score': float(s)}
                    for t, s in zip(phonetic_times, phonetic_scores)
                ]
            },
            'stats': {
                'reference_duration': duration,
                'input_duration': len(input_features['melody']) * 0.05,
                'n_melody_detections': len(melody_scores),
                'n_phonetic_detections': len(phonetic_scores)
            }
        }
        
        return results
        
    except Exception as e:
        logging.error(f"Error scoring audio: {e}")
        logging.exception("Full traceback:")
        raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Score audio against reference")
    parser.add_argument("reference", help="Path to reference audio file")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Validate files exist
        reference_path = Path(args.reference)
        input_path = Path(args.input)
        
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference file not found: {reference_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Score the audio
        results = score_audio(str(reference_path), str(input_path))
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results written to {output_path}")
        else:
            # Pretty print to console
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        logging.error(f"Error: {e}")
        raise SystemExit(1)

if __name__ == '__main__':
    main()
