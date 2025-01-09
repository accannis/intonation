"""
Visualization window for displaying scores and performance metrics
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QTextEdit, QSizePolicy, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np
import logging
from collections import deque
from typing import List, Optional, Dict
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
        self.melody_scores = deque(maxlen=1000)
        self.melody_times = deque(maxlen=1000)
        self.phonetic_scores = deque(maxlen=1000)
        self.phonetic_times = deque(maxlen=1000)
        self.total_scores = deque(maxlen=1000)
        self.total_times = deque(maxlen=1000)
        
        # Create central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(5)  # Reduce spacing between elements
        
        # Create top info section (session info + metrics)
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)  # Spacing between info and metrics
        
        # Create session info group
        info_group = QGroupBox("Session Info")
        info_group.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                padding-top: 8px;
                margin-top: 0px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
        """)
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)  # Minimal spacing between labels
        info_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        
        self.ref_file_label = QLabel("Reference Audio: Not set")
        self.input_source_label = QLabel("Input Source: Not set")
        for label in [self.ref_file_label, self.input_source_label]:
            label.setStyleSheet("font-size: 11px;")
        
        info_layout.addWidget(self.ref_file_label)
        info_layout.addWidget(self.input_source_label)
        info_group.setLayout(info_layout)
        top_layout.addWidget(info_group)
        
        # Create metrics group
        metrics_group = QGroupBox("Performance Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                padding-top: 8px;
                margin-top: 0px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
        """)
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(2)  # Minimal spacing between metrics
        metrics_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        self.metrics_labels = {}
        metrics_group.setLayout(metrics_layout)
        top_layout.addWidget(metrics_group)
        
        # Add top section to main layout
        main_layout.addLayout(top_layout)
        
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
        
    def set_session_info(self, reference_file: str, input_source: str, input_file: Optional[str] = None):
        """Update session info display
        
        Args:
            reference_file: Path to reference audio file
            input_source: Input source (microphone or file)
            input_file: Optional path to input audio file
        """
        import os
        self.ref_file_label.setText(f"Reference Audio: {os.path.basename(reference_file)}")
        
        input_text = f"Input Source: {input_source.title()}"
        if input_file:
            input_text += f" ({os.path.basename(input_file)})"
        self.input_source_label.setText(input_text)
        
    def update_scores(self, total_score: float, melody_score: float, phonetic_score: float, 
                     waveform: np.ndarray, duration: float, timestamp: float = 0):
        """Update all visualization elements with new scores
        
        Args:
            total_score: Current overall score (0-100)
            melody_score: Current melody matching score (0-100)
            phonetic_score: Current phonetic matching score (0-100)
            waveform: Audio waveform data
            duration: Duration of the current audio chunk
            timestamp: Time offset for this chunk
        """
        try:
            # Update score labels
            self.total_score_label.setText(f"Total Score: {total_score:.1f}")
            self.melody_score_label.setText(f"Melody Score: {melody_score:.1f}")
            self.phonetic_score_label.setText(f"Phonetic Score: {phonetic_score:.1f}")
            
            # Add new scores to history
            current_time = timestamp + duration
            self.total_scores.append(total_score)
            self.total_times.append(current_time)
            self.melody_scores.append(melody_score)
            self.melody_times.append(current_time)
            self.phonetic_scores.append(phonetic_score)
            self.phonetic_times.append(current_time)
            
            # Update plot ranges if needed
            if current_time > self.duration:
                self.duration = current_time * 1.5  # Add some extra space
                for plot in [self.total_score_plot, self.melody_score_plot, self.phonetic_score_plot, self.waveform_plot]:
                    plot.setXRange(0, self.duration)
            
            # Update score plots
            self.total_score_curve.setData(list(self.total_times), list(self.total_scores))
            self.melody_score_curve.setData(list(self.melody_times), list(self.melody_scores))
            self.phonetic_score_curve.setData(list(self.phonetic_times), list(self.phonetic_scores))
            
            # Update waveform plot
            if len(waveform) > 0:
                # Create time points for waveform
                times = np.linspace(timestamp, timestamp + duration, len(waveform))
                self.waveform_curve.setData(times, waveform)
                
        except Exception as e:
            logging.error(f"Error updating visualization: {e}")
            logging.exception("Full traceback:")
            
    def update_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics display
        
        Args:
            metrics: Dict mapping stage names to durations in seconds
        """
        # Create or update labels for each metric
        for stage, duration in metrics.items():
            if stage not in self.metrics_labels:
                label = QLabel()
                label.setStyleSheet("font-size: 11px;")  # Small font for metrics
                self.metrics_labels[stage] = label
                self.centralWidget().layout().itemAt(0).itemAt(1).widget().layout().addWidget(label)
            
            # Update label text
            self.metrics_labels[stage].setText(f"{stage}: {duration:.3f}s")
