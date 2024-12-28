# Singing Score ML Algorithm

An ML-based singing evaluation system that scores vocal performances based on melody and lyric accuracy.

## Features
- Real-time singing evaluation
- Melody matching (key-independent)
- Lyric pronunciation scoring
- Optimized for embedded devices

## Requirements
- Python 3.8+
- See requirements.txt for dependencies

## Project Structure
```
├── src/
│   ├── feature_extraction/    # Audio processing and feature extraction
│   ├── melody_matching/       # DTW-based melody comparison
│   ├── lyric_matching/        # Phonetic alignment and matching
│   ├── scoring/              # Score computation
│   └── utils/                # Helper functions
├── models/                   # Trained models and weights
├── data/                    # Reference songs and test data
└── tests/                   # Unit tests
```

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
(Documentation will be added as components are implemented)
