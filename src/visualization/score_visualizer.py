"""
Visualization window for displaying scores and performance metrics
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QTextEdit, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np
import logging
from collections import deque
from typing import List, Optional
from scipy.interpolate import interp1d

# Configure pyqtgraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class ScoreVisualizer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Singing Score Visualization")
        self.setGeometry(100, 100, 1000, 800)
        
        # Initialize data storage
        self.duration = None
        self.melody_scores = None
        self.melody_times = None
        self.phonetic_scores = None
        self.phonetic_times = None
        self.total_scores = None
        self.total_times = None
        self.waveform_data = None
        self.waveform_times = None
        
        # Create central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Create score labels
        scores_layout = QHBoxLayout()
        
        self.total_score_label = QLabel("Total Score: 0.0")
        self.total_score_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: #2c3e50;")
        scores_layout.addWidget(self.total_score_label)
        
        self.melody_score_label = QLabel("Melody Score: 0.0")
        self.melody_score_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: #e74c3c;")
        scores_layout.addWidget(self.melody_score_label)
        
        self.phonetic_score_label = QLabel("Phonetic Score: 0.0")
        self.phonetic_score_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: #2980b9;")
        scores_layout.addWidget(self.phonetic_score_label)
        
        main_layout.addLayout(scores_layout)
        
        # Create plot widgets
        self.plots_layout = QVBoxLayout()
        
        # Total score plot
        self.total_score_plot = pg.PlotWidget(title="Total Score")
        self.total_score_plot.setBackground('w')
        self.total_score_plot.showGrid(x=True, y=True)
        self.total_score_curve = self.total_score_plot.plot(pen=pg.mkPen(color='#2c3e50', width=2))
        self.plots_layout.addWidget(self.total_score_plot)
        
        # Melody score plot
        self.melody_score_plot = pg.PlotWidget(title="Melody Score")
        self.melody_score_plot.setBackground('w')
        self.melody_score_plot.showGrid(x=True, y=True)
        self.melody_score_curve = self.melody_score_plot.plot(pen=pg.mkPen(color='#e74c3c', width=2))
        self.plots_layout.addWidget(self.melody_score_plot)
        
        # Phonetic score plot
        self.phonetic_score_plot = pg.PlotWidget(title="Phonetic Score")
        self.phonetic_score_plot.setBackground('w')
        self.phonetic_score_plot.showGrid(x=True, y=True)
        self.phonetic_score_curve = self.phonetic_score_plot.plot(pen=pg.mkPen(color='#2980b9', width=2))
        self.plots_layout.addWidget(self.phonetic_score_plot)
        
        # Waveform plot
        self.waveform_plot = pg.PlotWidget(title="Audio Waveform")
        self.waveform_plot.setBackground('w')
        self.waveform_plot.showGrid(x=True, y=True)
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen(color='#27ae60', width=1))
        self.plots_layout.addWidget(self.waveform_plot)
        
        main_layout.addLayout(self.plots_layout)
        
    def initialize(self, duration: float):
        """Initialize the visualizer with the session duration
        
        Args:
            duration: Total duration of the session in seconds
        """
        self.duration = duration
        
        # Set up plot ranges
        for plot in [self.total_score_plot, self.melody_score_plot, self.phonetic_score_plot]:
            plot.setXRange(0, duration)
            plot.setYRange(0, 100)
            
        self.waveform_plot.setXRange(0, duration)
        self.waveform_plot.setYRange(-1, 1)
        
        logging.info(f"Visualizer initialized with duration: {duration:.2f}s")
        
    def update_scores(self, 
                     total_score: float,
                     melody_score: float,
                     phonetic_score: float,
                     waveform: np.ndarray,
                     melody_scores: Optional[List[float]] = None,
                     melody_times: Optional[List[float]] = None,
                     phonetic_scores: Optional[List[float]] = None,
                     phonetic_times: Optional[List[float]] = None):
        """Update all visualization elements with new scores
        
        Args:
            total_score: Current overall score (0-100)
            melody_score: Current melody matching score (0-100)
            phonetic_score: Current phonetic matching score (0-100)
            waveform: Audio waveform data
            melody_scores: List of melody scores over time
            melody_times: List of time points for melody scores
            phonetic_scores: List of phonetic scores over time
            phonetic_times: List of time points for phonetic scores
        """
        try:
            # Update score labels with current scores
            self.total_score_label.setText(f"Total Score: {total_score:.1f}")
            self.melody_score_label.setText(f"Melody Score: {melody_score:.1f}")
            self.phonetic_score_label.setText(f"Phonetic Score: {phonetic_score:.1f}")
            
            # Generate time points for the waveform
            if self.duration is not None:
                # Create time points for original waveform
                original_times = np.linspace(0, self.duration, len(waveform))
                
                # Create time points for display (use window width)
                display_points = self.waveform_plot.width()
                display_times = np.linspace(0, self.duration, display_points)
                
                # Resample waveform to match display points
                # Use max of absolute values to preserve peaks
                points_per_pixel = len(waveform) // display_points
                if points_per_pixel > 1:
                    # Reshape to get chunks for each pixel
                    n_chunks = display_points * points_per_pixel
                    chunks = waveform[:n_chunks].reshape(display_points, points_per_pixel)
                    display_waveform = np.max(np.abs(chunks), axis=1) * np.sign(np.mean(chunks, axis=1))
                else:
                    # If we have fewer points than pixels, interpolate
                    interp_func = interp1d(original_times, waveform, kind='linear')
                    display_waveform = interp_func(display_times)
                
                # Update waveform plot
                self.waveform_curve.setData(display_times, display_waveform)
                
                # Update melody score plot if we have time series data
                if melody_scores is not None and melody_times is not None and len(melody_scores) > 0:
                    self.melody_scores = np.array(melody_scores)
                    self.melody_times = np.array(melody_times)
                    self.melody_score_curve.setData(self.melody_times, self.melody_scores)
                    logging.info(f"Updated melody plot with {len(melody_scores)} points")
                
                # Update phonetic score plot if we have time series data
                if phonetic_scores is not None and phonetic_times is not None and len(phonetic_scores) > 0:
                    self.phonetic_scores = np.array(phonetic_scores)
                    self.phonetic_times = np.array(phonetic_times)
                    self.phonetic_score_curve.setData(self.phonetic_times, self.phonetic_scores)
                    logging.info(f"Updated phonetic plot with {len(phonetic_scores)} points")
                
                # Update total score plot
                # If we have both melody and phonetic time series, interpolate to get total score
                if (self.melody_scores is not None and self.phonetic_scores is not None and 
                    self.melody_times is not None and self.phonetic_times is not None and
                    len(self.melody_scores) > 0 and len(self.phonetic_scores) > 0):
                    
                    # Use the finer time resolution
                    if len(self.melody_times) > len(self.phonetic_times):
                        self.total_times = self.melody_times
                        interpolated_phonetic = np.interp(
                            self.total_times, 
                            self.phonetic_times, 
                            self.phonetic_scores
                        )
                        self.total_scores = 0.6 * self.melody_scores + 0.4 * interpolated_phonetic
                    else:
                        self.total_times = self.phonetic_times
                        interpolated_melody = np.interp(
                            self.total_times,
                            self.melody_times,
                            self.melody_scores
                        )
                        self.total_scores = 0.6 * interpolated_melody + 0.4 * self.phonetic_scores
                    
                    self.total_score_curve.setData(self.total_times, self.total_scores)
                    logging.info(f"Updated total score plot with {len(self.total_scores)} points")
                
                logging.info(
                    f"Visualization updated - "
                    f"Duration: {self.duration:.1f}s, "
                    f"Original waveform: {len(waveform)} points, "
                    f"Display waveform: {len(display_waveform)} points, "
                    f"Melody points: {len(melody_scores) if melody_scores else 0}, "
                    f"Phonetic points: {len(phonetic_scores) if phonetic_scores else 0}"
                )
            
        except Exception as e:
            logging.error(f"Error updating visualization: {e}")
            logging.exception("Full traceback:")
