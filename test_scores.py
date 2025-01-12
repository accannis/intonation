import logging
import json
import sys
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

def main():
    print("\nRunning scoring tests...")
    print("=" * 80)

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
            "melody_range": (80, 100)  # Professional performance should score high
        },
        {
            "ref": "healingincarnation.wav",
            "input": "../Audio/SungWell.wav", 
            "description": "User singing - good melody and lyrics",
            "melody_range": (80, 100)  # Good performance should score high
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
            "melody_range": (70, 100)  # Good melody should score high regardless of words
        },
        {
            "ref": "healingincarnation.wav",
            "input": "../Audio/WrongLyricsAndMelody.wav",
            "description": "Different song - wrong melody and lyrics",
            "melody_range": (0, 20)  # Wrong melody should score very low
        }
    ]

    # Run tests
    all_passed = True
    for test in test_cases:
        success = run_test(
            test["ref"],
            test["input"],
            test["description"],
            test.get("melody_range"),
            test.get("phonetic_range"),
            enable_melody=True,  
            enable_lyrics=True
        )
        all_passed = all_passed and success

    print("\n" + "=" * 80)
    if all_passed:
        print("All tests passed ")
    else:
        print("Some tests failed ")

if __name__ == "__main__":
    main()
