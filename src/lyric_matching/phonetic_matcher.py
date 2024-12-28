import numpy as np
from typing import List, Tuple
import whisper
import re
from Levenshtein import distance
import logging
import soundfile as sf
import tempfile
import os

class PhoneticMatcher:
    def __init__(self):
        # Initialize Whisper for speech recognition
        logging.info("Loading Whisper model...")
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(
            "base",
            device=self.device,
            download_root=None,
            in_memory=True
        )
        logging.info(f"Whisper model loaded on {self.device}")
        
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to tokens for matching
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Tokenize text into words using regex and convert to lowercase
        return re.findall(r'\b\w+\b', text.lower())
        
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio to text using Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        options = dict(
            language='en',
            task='transcribe',
            without_timestamps=True,
            fp16=False
        )
        try:
            result = self.whisper_model.transcribe(audio_path, **options)
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return ""
        
    def audio_to_phonemes(self, audio_data: np.ndarray, sample_rate: int) -> List[str]:
        """
        Convert audio data to phonemes using Whisper
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of phonemes
        """
        # Ensure audio data is long enough for processing (at least 1 second)
        min_samples = sample_rate  # 1 second
        if len(audio_data) < min_samples:
            # Pad with zeros if too short
            padding = np.zeros(min_samples - len(audio_data))
            audio_data = np.concatenate([audio_data, padding])
        elif len(audio_data) > sample_rate * 30:  # Limit to 30 seconds
            audio_data = audio_data[:sample_rate * 30]
            
        # Ensure audio data is normalized to [-1, 1]
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        # Resample to target length for Whisper
        target_length = 100  # Match the pitch sequence length
        if len(audio_data) > target_length:
            indices = np.linspace(0, len(audio_data)-1, target_length, dtype=int)
            audio_data = audio_data[indices]
        elif len(audio_data) < target_length:
            padded = np.zeros(target_length)
            padded[:len(audio_data)] = audio_data
            audio_data = padded
        
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            temp_path = temp_file.name
            
        try:
            # Transcribe the temporary audio file
            transcribed_text = self.transcribe_audio(temp_path)
            # Convert transcribed text to phonemes
            return self.text_to_phonemes(transcribed_text)
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    def calculate_similarity(self, reference_tokens: List[str], 
                           sung_tokens: List[str]) -> float:
        """
        Calculate similarity between reference and sung lyrics
        
        Args:
            reference_tokens: Reference lyrics tokens
            sung_tokens: Sung lyrics tokens
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use Levenshtein distance for similarity
        max_len = max(len(reference_tokens), len(sung_tokens))
        if max_len == 0:
            return 1.0
            
        distance_score = distance(reference_tokens, sung_tokens)
        return 1.0 - (distance_score / max_len)
        
    def match_lyrics(self, reference_text: str, sung_text: str) -> Tuple[float, str]:
        """
        Match reference lyrics with sung lyrics
        
        Args:
            reference_text: Reference lyrics
            sung_text: Transcribed lyrics from singing
            
        Returns:
            Tuple of (similarity score, feedback)
        """
        # Convert both texts to tokens
        reference_tokens = self.text_to_phonemes(reference_text)
        sung_tokens = self.text_to_phonemes(sung_text)
        
        # Calculate similarity
        similarity = self.calculate_similarity(reference_tokens, sung_tokens)
        
        # Generate feedback
        if similarity >= 0.9:
            feedback = "Excellent pronunciation!"
        elif similarity >= 0.7:
            feedback = "Good pronunciation, keep practicing!"
        else:
            feedback = "Try to pronounce the words more clearly."
            
        return similarity, feedback
