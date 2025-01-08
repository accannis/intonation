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

# Configure pyqtgraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class ScoreVisualizer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session")
        self.setGeometry(100, 100, 800, 900)
        
        # Initialize data storage
        self.start_time = None
        self.timestamps = []
        self.melody_scores = []
        self.phonetic_scores = []
        self.total_scores = []
        self.waveform_data = []
        self.waveform_times = []
        
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
        
        # Create plots with consistent styling
        self.waveform_plot = self._create_plot("Audio Waveform", "Time (s)", "Amplitude")
        self.total_plot = self._create_plot("Total Score", "Time (s)", "Score")
        self.melody_plot = self._create_plot("Melody Score", "Time (s)", "Score")
        self.phonetic_plot = self._create_plot("Phonetic Score", "Time (s)", "Score")
        
        # Add plots to layout with size policies
        for plot in [self.waveform_plot, self.total_plot, self.melody_plot, self.phonetic_plot]:
            plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            main_layout.addWidget(plot)
        
        # Give waveform more height
        self.waveform_plot.setMinimumHeight(200)
        
        # Create feedback display
        self.feedback_label = QLabel("Feedback:")
        self.feedback_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #34495e;")
        main_layout.addWidget(self.feedback_label)
        
        self.feedback_display = QTextEdit()
        self.feedback_display.setReadOnly(True)
        self.feedback_display.setMaximumHeight(100)
        self.feedback_display.setStyleSheet("""
            QTextEdit {
                font-size: 14pt;
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                color: #212529;
            }
        """)
        main_layout.addWidget(self.feedback_display)
        
        # Create plot curves with consistent colors
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen(color='b', width=1))
        self.total_curve = self.total_plot.plot(pen=pg.mkPen(color='r', width=2))
        self.melody_curve = self.melody_plot.plot(pen=pg.mkPen(color='g', width=2))
        self.phonetic_curve = self.phonetic_plot.plot(pen=pg.mkPen(color='b', width=2))
        
        # Start update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(50)  # Update every 50ms
        
    def _create_plot(self, title, x_label, y_label):
        """Create a pyqtgraph plot with consistent styling"""
        plot = pg.PlotWidget()
        plot.setBackground('w')
        plot.setTitle(title, color='k', size='12pt')
        plot.setLabel('left', y_label, color='k')
        plot.setLabel('bottom', x_label, color='k')
        plot.showGrid(x=True, y=True)
        
        # Style the axes
        plot.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        plot.getAxis('left').setPen(pg.mkPen(color='k', width=1))
        plot.getAxis('bottom').setTextPen(pg.mkPen(color='k'))
        plot.getAxis('left').setTextPen(pg.mkPen(color='k'))
        
        # Set y-axis range
        if 'Score' in y_label:
            plot.setYRange(0, 100)
        else:
            plot.setYRange(-1, 1)  # For waveform
            
        return plot
        
    def update_plots(self):
        """Update all plots to show entire session duration"""
        try:
            if not self.timestamps or self.start_time is None:
                return
                
            # Calculate relative times from session start
            rel_times = [t - self.start_time for t in self.timestamps]
            rel_waveform_times = [t - self.start_time for t in self.waveform_times]
            
            # Update all plots to show entire session
            duration = rel_times[-1]
            for plot in [self.waveform_plot, self.total_plot, self.melody_plot, self.phonetic_plot]:
                plot.setXRange(0, duration)
                
            # Update curves with relative times
            self.total_curve.setData(rel_times, self.total_scores)
            self.melody_curve.setData(rel_times, self.melody_scores)
            self.phonetic_curve.setData(rel_times, self.phonetic_scores)
            self.waveform_curve.setData(rel_waveform_times, self.waveform_data)
                
        except Exception as e:
            logging.error(f"Error updating plots: {e}")
            logging.exception("Full traceback:")
            
    def update_display(self, features, timestamp, audio_data=None):
        """Update display with new data"""
        try:
            if not features or 'total_score' not in features:
                return
                
            # Initialize start time if not set
            if self.start_time is None:
                self.start_time = timestamp
                
            # Update scores
            melody_score = features.get('melody_score', 0)
            phonetic_score = features.get('phonetic_score', 0)
            total_score = features.get('total_score', 0)
            
            # Update labels
            self.melody_score_label.setText(f"Melody Score: {melody_score:.1f}")
            self.phonetic_score_label.setText(f"Phonetic Score: {phonetic_score:.1f}")
            self.total_score_label.setText(f"Total Score: {total_score:.1f}")
            
            # Store scores
            self.timestamps.append(timestamp)
            self.melody_scores.append(melody_score)
            self.phonetic_scores.append(phonetic_score)
            self.total_scores.append(total_score)
            
            # Update waveform data if provided
            if audio_data is not None:
                # Create time points for new audio data
                sample_rate = 44100
                num_samples = len(audio_data)
                new_times = np.linspace(timestamp, timestamp + num_samples/sample_rate, num_samples)
                
                # Add new data points
                self.waveform_data.extend(audio_data.tolist())
                self.waveform_times.extend(new_times.tolist())
            
            # Update feedback if available
            if 'feedback' in features:
                self.feedback_display.setText(features['feedback'])
                
        except Exception as e:
            logging.error(f"Error updating display: {e}")
            logging.exception("Full traceback:")
