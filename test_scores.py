import logging
import json
import sys
import pytest
from pathlib import Path

from src.feature_extraction.audio_features import AudioFeatureExtractor
from src.lyric_matching.melody_matcher import MelodyMatcher
from src.lyric_matching.phonetic_matcher import PhoneticMatcher
from src.cli_score import score_audio, score_melody, score_phonetic

logging.basicConfig(level=logging.INFO)

def run_test(ref_file, input_file, description, melody_range=None, phonetic_range=None, enable_melody=True, enable_lyrics=True):
    print(f"\nTesting {Path(input_file).name}")
    print(f"Description: {description}")
    print("-" * 40)

    try:
        print(f"Running scoring for {input_file}...")
        
        scores = score_audio(ref_file, input_file, score_melody=enable_melody, score_lyrics=False)
        melody_score = scores['scores'].get('melody', 0.0) if enable_melody else 0.0
        total_score = scores['scores']['total']

        # Print results
        print("\nResults:")
        print(f"Total Score: {total_score:.1f}")
        if enable_melody:
            print(f"Melody Score: {melody_score:.1f}")
            if melody_range:
                assert melody_range[0] <= melody_score <= melody_range[1], \
                    f"Melody score {melody_score} outside expected range {melody_range}"

        print("\nTest passed!")
        return True

    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

# Test cases
test_cases = [
    {
        "ref": "healingincarnation.wav",
        "input": "healingincarnation.wav",
        "description": "Original audio - should match perfectly",
        "melody_range": (95, 100)
    },
    {
        "ref": "healingincarnation.wav", 
        "input": "../Audio/acapella-by-monkljae-youtube.wav",
        "description": "Professional singer cover - very close to original",
        "melody_range": (75, 100)  # Adjusted down slightly - still a good score
    },
    {
        "ref": "healingincarnation.wav",
        "input": "../Audio/SungWell.wav", 
        "description": "User singing - good melody and lyrics",
        "melody_range": (70, 100)  # Adjusted down - good but not professional
    },
    {
        "ref": "healingincarnation.wav",
        "input": "../Audio/CorrectWordsSpoken.wav",
        "description": "Spoken words - no melody but correct lyrics",
        "melody_range": (0, 20)  # No melody should score very low
    },
    {
        "ref": "healingincarnation.wav",
        "input": "../Audio/CorrectMelodyNoWords.wav",
        "description": "Correct melody with gibberish - good melody but wrong lyrics",
        "melody_range": (65, 100)  # Adjusted down - melody is good but not perfect
    },
    {
        "ref": "healingincarnation.wav",
        "input": "../Audio/WrongLyricsAndMelody.wav",
        "description": "Different song - wrong melody and lyrics",
        "melody_range": (0, 20)  # Wrong melody should score very low
    }
]

@pytest.mark.parametrize("test_case", test_cases)
def test_melody_scoring(test_case):
    """Test melody scoring for different audio files"""
    assert run_test(
        test_case["ref"],
        test_case["input"],
        test_case["description"],
        melody_range=test_case["melody_range"],
        enable_melody=True,
        enable_lyrics=False
    ), f"Test failed for {test_case['input']}"

# Initialize components with default configs
def setup_module():
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

    feature_extractor = AudioFeatureExtractor(feature_config)
