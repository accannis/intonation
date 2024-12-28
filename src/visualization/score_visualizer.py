import pyqtgraph as pg
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
                           QComboBox, QPushButton, QFileDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import numpy as np
from typing import Dict, List
import colorsys
import logging
from collections import deque
import time

class ScoreVisualizer(QMainWindow):
    # Add signal for audio source change
    audio_source_changed = pyqtSignal(str, str)  # source_type, file_path
    
    def __init__(self):
        """Initialize the score visualizer"""
        logging.info("Initializing ScoreVisualizer...")
        super().__init__()
        
        # Initialize data arrays with deques for unlimited history
        logging.info("Initializing data arrays...")
        self.start_time = time.time()
        self.timestamps = deque()
        self.melody_scores = deque()
        self.lyric_scores = deque()
        self.total_scores = deque()
        
        # Store feedback messages
        self.feedback_history: List[str] = []
        
        # Setup UI components
        logging.info("Setting up UI components...")
        self.setup_ui()
        
        # Update timer
        logging.info("Setting up update timer...")
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # 50ms refresh rate
        
        logging.info("ScoreVisualizer initialization complete")
        
    def setup_ui(self):
        """Initialize the UI components"""
        self.setWindowTitle("Singing Score Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Add audio source selector
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Audio Source:"))
        
        self.source_selector = QComboBox()
        self.source_selector.addItems(["Audio Input", "Audio File"])
        self.source_selector.currentTextChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_selector)
        
        self.file_button = QPushButton("Select File...")
        self.file_button.clicked.connect(self._on_file_button_clicked)
        self.file_button.setVisible(False)
        source_layout.addWidget(self.file_button)
        
        source_layout.addStretch()
        layout.addLayout(source_layout)
        
        # Create tabs for different views
        self.tabs = QTabWidget()
        
        # Main scoring tab
        scoring_tab = QWidget()
        scoring_layout = QVBoxLayout(scoring_tab)
        
        # Create score indicators
        score_layout = QHBoxLayout()
        
        # Total score
        self.total_score_label = QLabel("Total Score: 0")
        self.total_score_label.setStyleSheet(
            "QLabel { font-size: 24pt; font-weight: bold; }"
        )
        score_layout.addWidget(self.total_score_label)
        
        # Melody score
        self.melody_score_label = QLabel("Melody Score: 0")
        self.melody_score_label.setStyleSheet(
            "QLabel { font-size: 24pt; font-weight: bold; }"
        )
        score_layout.addWidget(self.melody_score_label)
        
        # Lyrics score
        self.lyrics_score_label = QLabel("Lyrics Score: 0")
        self.lyrics_score_label.setStyleSheet(
            "QLabel { font-size: 24pt; font-weight: bold; }"
        )
        score_layout.addWidget(self.lyrics_score_label)
        
        scoring_layout.addLayout(score_layout)
        
        # Create progress bars with labels
        progress_layout = QVBoxLayout()
        metrics_layout = QHBoxLayout()
        
        # Create detailed metrics widgets
        self.metrics_widgets = {}
        for metric in ['Pitch Accuracy', 'Rhythm', 'Pronunciation', 'Breath Control']:
            widget = QWidget()
            metric_layout = QVBoxLayout(widget)
            
            label = QLabel(metric)
            progress = self._create_progress_bar()
            value_label = QLabel("0%")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            metric_layout.addWidget(label)
            metric_layout.addWidget(progress)
            metric_layout.addWidget(value_label)
            
            self.metrics_widgets[metric] = {
                'progress': progress,
                'value': value_label
            }
            
            metrics_layout.addWidget(widget)
        
        progress_layout.addLayout(metrics_layout)
        scoring_layout.addLayout(progress_layout)
        
        # Create plots
        plots_layout = QHBoxLayout()
        
        # Score history plot
        self.score_plot = pg.PlotWidget()
        self.score_plot.setBackground('w')
        self.score_plot.setTitle("Score History", color='k')
        self.score_plot.setLabel('left', 'Score', color='k')
        self.score_plot.setLabel('bottom', 'Time (s)', color='k')
        self.score_plot.setYRange(0, 100)
        self.score_plot.showGrid(x=True, y=True)
        
        # Create plot lines
        self.total_line = self.score_plot.plot(pen=pg.mkPen(color='b', width=2))
        self.melody_line = self.score_plot.plot(pen=pg.mkPen(color='g', width=2))
        self.lyric_line = self.score_plot.plot(pen=pg.mkPen(color='r', width=2))
        
        plots_layout.addWidget(self.score_plot)
        scoring_layout.addLayout(plots_layout)
        
        # Add feedback history
        self.feedback_table = QTableWidget()
        self.feedback_table.setColumnCount(1)
        self.feedback_table.setHorizontalHeaderLabels(["Feedback"])
        self.feedback_table.horizontalHeader().setStretchLastSection(True)
        scoring_layout.addWidget(self.feedback_table)
        
        self.tabs.addTab(scoring_tab, "Scoring")
        layout.addWidget(self.tabs)
        
    def show(self):
        """Show the window"""
        super().show()
        
    def _create_progress_bar(self):
        """Create a styled progress bar"""
        progress = QProgressBar()
        progress.setMinimum(0)
        progress.setMaximum(100)
        progress.setTextVisible(True)
        progress.setFormat("%v%")
        return progress
        
    def _get_color_for_score(self, score: float):
        """Get color based on score value"""
        if score < 0 or score > 100:
            return QColor(128, 128, 128)  # Gray for invalid scores
            
        # Use HSV color space: red (0) for low scores, green (120) for high scores
        hue = score * 1.2  # 120 degrees = green
        rgb = colorsys.hsv_to_rgb(hue/360, 1.0, 1.0)
        return QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
    def update_data(self, scores: Dict[str, float], feedback: str, detailed_metrics: Dict = None):
        """Update the visualization with new data"""
        # Add current time to history
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        
        # Add new scores
        self.melody_scores.append(scores.get('melody', 0))
        self.lyric_scores.append(scores.get('lyrics', 0))
        self.total_scores.append(scores.get('total', 0))
        
        # Update labels
        self.total_score_label.setText(f"Total Score: {scores.get('total', 0):.1f}")
        self.melody_score_label.setText(f"Melody Score: {scores.get('melody', 0):.1f}")
        self.lyrics_score_label.setText(f"Lyrics Score: {scores.get('lyrics', 0):.1f}")
        
        # Update detailed metrics
        if detailed_metrics:
            for metric, value in detailed_metrics.items():
                if metric in self.metrics_widgets:
                    self.metrics_widgets[metric]['progress'].setValue(int(value))
                    self.metrics_widgets[metric]['value'].setText(f"{value:.1f}%")
                    
        # Add feedback to history
        if feedback:
            row = self.feedback_table.rowCount()
            self.feedback_table.insertRow(row)
            self.feedback_table.setItem(row, 0, QTableWidgetItem(feedback))
            self.feedback_table.scrollToBottom()
            
    def update_plots(self):
        """Update the plot lines"""
        if not self.timestamps:
            return
            
        # Convert deques to numpy arrays for plotting
        timestamps = np.array(self.timestamps)
        total_scores = np.array(self.total_scores)
        melody_scores = np.array(self.melody_scores)
        lyric_scores = np.array(self.lyric_scores)
        
        # Update plot range
        self.score_plot.setXRange(0, max(timestamps))
        
        # Update plot lines
        self.total_line.setData(timestamps, total_scores)
        self.melody_line.setData(timestamps, melody_scores)
        self.lyric_line.setData(timestamps, lyric_scores)
        
    def _on_source_changed(self, source: str):
        """Handle audio source selection change"""
        if source == "Audio Input":
            self.file_button.setVisible(False)
            self.audio_source_changed.emit("input", "")
        else:
            self.file_button.setVisible(True)
            
    def _on_file_button_clicked(self):
        """Handle file selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "WAV Files (*.wav)"
        )
        if file_path:
            self.audio_source_changed.emit("file", file_path)
