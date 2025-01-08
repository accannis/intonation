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

3. **Scoring System**
   - `MelodyMatcher`: DTW-based melody comparison (60% weight)
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
│   ├── phonetic_matcher.py
│   └── lyric_provider.py
├── scoring/
│   └── score_calculator.py
├── visualization/
│   ├── score_visualizer.py
│   └── setup_window.py
└── main.py
```

### Current Features
1. Support for both file and microphone input
2. Real-time scoring and visualization
3. Full session duration display
4. Vocal separation for better scoring
5. Setup window for input selection

### Technical Specs
1. Audio Processing:
   - Sample Rate: 44.1kHz
   - Chunk Size: Configurable (default matches Demucs)
   - Format: WAV files supported

2. Scoring:
   - Melody Score: 0-100 scale
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
- Demucs: Vocal separation

### Recent Changes
1. Modularized session management
2. Improved visualization timing
3. Separated reference/input processing
4. Enhanced error handling

### Next Steps/TODOs
1. Performance optimization
2. Enhanced error feedback
3. Support for more audio formats
4. Session recording/playback
5. More detailed scoring feedback

Last Updated: 2025-01-08
