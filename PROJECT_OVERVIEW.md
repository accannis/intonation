# Windsurfing Project Overview

### Project Overview
A real-time singing evaluation system that scores users' singing performance by comparing it against a reference audio track.

### Core Components

1. **Audio Processing**
   - `AudioFileProcessor`: Handles audio file loading and playback
   - `MicrophoneProcessor`: Handles real-time microphone input
   - `VocalSeparator`: Uses Demucs model to separate vocals from music

2. **Feature Extraction**
   - `AudioFeatureExtractor`: Extracts MFCC, mel spectrograms, and pitch features
   - Processes both reference and input audio in chunks
   - Caches extracted features for performance
   - Configurable feature parameters

3. **Scoring System**
   - `MelodyMatcher`: DTW-based melody comparison (60% weight)
     - Minimum 10 seconds (200 frames) of voiced audio
     - At least 30% of frames must be voiced
     - High pitch confidence (≥0.95)
     - Scores pushed toward extremes using sigmoid (k=20)
   - `PhoneticMatcher`: Phonetic/lyric matching (40% weight)
   - `ScoreCalculator`: Combines scores with configurable weights

4. **Visualization**
   - `ScoreVisualizer`: Real-time display of:
     - Total score
     - Melody score
     - Phonetic score
     - Audio waveform
   - All plots show full session duration
   - White background with clear axes

5. **Session Management**
   - `ScoringSession`: Manages a complete scoring session
     - Initializes all components
     - Processes reference audio
     - Handles real-time scoring
     - Manages visualization
   - Separate processors for reference and input audio

### Key Files
```
src/
├── audio_processing/
│   ├── file_processor.py
│   ├── mic_processor.py
│   └── scoring_session.py
├── feature_extraction/
│   └── audio_features.py
├── melody_matching/
│   └── dtw_matcher.py
├── lyric_matching/
│   ├── melody_matcher.py  # New location for melody matching
│   └── phonetic_matcher.py
├── scoring/
│   └── score_calculator.py
├── visualization/
│   ├── score_visualizer.py
│   └── setup_window.py
├── cli_score.py  # New CLI interface
└── main.py

tests/
└── test_scores.py  # New test suite
```

### Current Features
1. Support for both file and microphone input
2. Real-time scoring and visualization
3. Full session duration display
4. Vocal separation for better scoring
5. Setup window for input selection
6. CLI interface for quick testing
   - Usage: `python cli_score.py reference.wav input.wav`
   - Options for melody-only or lyrics-only scoring
7. Comprehensive test suite with calibrated scoring:
   - Perfect matches (95-100): Original audio, professional covers
   - Good performances (80-100): Well-sung user performances
   - Poor performances (0-20): Spoken words, wrong melody/lyrics

### Technical Specs
1. Audio Processing:
   - Sample Rate: 44.1kHz
   - Chunk Size: Configurable (default matches Demucs)
   - Format: WAV files supported
   - Feature caching for performance

2. Scoring:
   - Melody Score: 0-100 scale
     - DTW for tempo-invariant matching
     - Strict voiced frame requirements
     - High confidence threshold
   - Phonetic Score: 0-100 scale
   - Total Score: Weighted average

3. Visualization:
   - Update Rate: 50ms
   - Time Range: Full session duration
   - Four synchronized plots

### Dependencies
- PyQt6: UI framework
- pyqtgraph: Real-time plotting
- torchaudio: Audio processing
- numpy: Numerical operations
- librosa: Audio feature extraction
- Demucs: Vocal separation

### Recent Changes
1. Modularized session management
2. Improved visualization timing
3. Separated reference/input processing
4. Enhanced error handling
5. Added CLI interface for testing
6. Implemented strict scoring criteria:
   - Minimum voiced frames requirement
   - Minimum voiced ratio check
   - Higher pitch confidence threshold
7. Added comprehensive test suite

### Next Steps/TODOs
1. Performance optimization
2. Enhanced error feedback
3. Support for more audio formats
4. Session recording/playback
5. More detailed scoring feedback
6. Real-time performance improvements
7. GUI interface enhancements

Last Updated: 2025-01-10
