#!/usr/bin/env python3

import sys
import os

# Add src directory to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from PyQt6.QtWidgets import QApplication
from src.visualization.score_visualizer import ScoreVisualizer
import numpy as np
import time

def main():
    app = QApplication(sys.argv)
    
    # Create and show the visualizer
    visualizer = ScoreVisualizer()
    visualizer.show()
    
    # For testing, let's generate some sample data
    def generate_test_data():
        t = time.time()
        melody = 50 + 30 * np.sin(t / 2)  # Oscillating melody score
        lyrics = 60 + 20 * np.cos(t / 3)   # Oscillating lyrics score
        total = (melody + lyrics) / 2       # Average for total score
        
        feedback = ("Keep singing!", ["Try to maintain pitch", "Good rhythm"])
        words = "test word" if t % 5 < 0.1 else ""  # Add word every 5 seconds
        
        visualizer.update_data(
            melody_score=melody,
            lyric_score=lyrics,
            total_score=total,
            timestamp=t,
            feedback=feedback,
            detected_words=words
        )
    
    # Create a timer to simulate real-time data
    from PyQt6.QtCore import QTimer
    timer = QTimer()
    timer.timeout.connect(generate_test_data)
    timer.start(50)  # Update every 50ms
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
