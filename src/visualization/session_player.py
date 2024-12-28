import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import h5py
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QSlider, QComboBox, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import pandas as pd

class SessionPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # Session data
        self.current_session: Optional[Dict] = None
        self.session_data: Optional[pd.DataFrame] = None
        self.audio_data: Optional[np.ndarray] = None
        self.current_frame = 0
        self.playing = False
        
        # Playback settings
        self.playback_speed = 1.0
        self.update_interval = 50  # ms
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def setup_ui(self):
        """Initialize the UI components"""
        self.setWindowTitle("Session Replay")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Session selection
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self.load_session)
        controls_layout.addWidget(QLabel("Session:"))
        controls_layout.addWidget(self.session_combo)
        
        # Playback controls
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_playback)
        controls_layout.addWidget(self.reset_button)
        
        # Speed control
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "2.0x", "4.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self.set_playback_speed)
        controls_layout.addWidget(QLabel("Speed:"))
        controls_layout.addWidget(self.speed_combo)
        
        layout.addLayout(controls_layout)
        
        # Create timeline slider
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(100)
        self.timeline.valueChanged.connect(self.seek_to_position)
        layout.addWidget(self.timeline)
        
        # Create visualization layout
        viz_layout = QHBoxLayout()
        
        # Score plot
        self.score_widget = pg.PlotWidget()
        self.score_widget.setBackground('w')
        self.score_widget.setTitle("Scores", color='k')
        self.score_widget.setLabel('left', 'Score', color='k')
        self.score_widget.setLabel('bottom', 'Time (s)', color='k')
        self.score_widget.showGrid(x=True, y=True)
        self.score_widget.setYRange(0, 100)
        
        # Create plot lines
        self.total_line = self.score_widget.plot([], [], pen=pg.mkPen(color='b', width=2))
        self.melody_line = self.score_widget.plot([], [], pen=pg.mkPen(color='g', width=2))
        self.lyric_line = self.score_widget.plot([], [], pen=pg.mkPen(color='r', width=2))
        
        viz_layout.addWidget(self.score_widget)
        
        # Metrics plot
        self.metrics_widget = pg.PlotWidget()
        self.metrics_widget.setBackground('w')
        self.metrics_widget.setTitle("Performance Metrics", color='k')
        self.metrics_widget.setLabel('left', 'Value', color='k')
        self.metrics_widget.setLabel('bottom', 'Time (s)', color='k')
        self.metrics_widget.showGrid(x=True, y=True)
        self.metrics_widget.setYRange(0, 1)
        
        # Create metrics lines
        self.metrics_lines = {}
        metrics = ['pitch_accuracy', 'rhythm_stability', 
                  'pronunciation_clarity', 'breath_control']
        colors = ['b', 'g', 'r', 'y']
        
        for metric, color in zip(metrics, colors):
            self.metrics_lines[metric] = self.metrics_widget.plot(
                [], [], 
                pen=pg.mkPen(color=color, width=2),
                name=metric.replace('_', ' ').title()
            )
            
        viz_layout.addWidget(self.metrics_widget)
        layout.addLayout(viz_layout)
        
        # Create feedback display
        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet(
            "QLabel { background-color: white; padding: 10px; border: 1px solid gray; }"
        )
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setMinimumHeight(100)
        layout.addWidget(self.feedback_label)
        
        # Refresh available sessions
        self.refresh_sessions()
        
    def refresh_sessions(self):
        """Load available session files"""
        session_dir = Path("session_history")
        if not session_dir.exists():
            return
            
        sessions = []
        for session_path in session_dir.glob("**/session_summary.json"):
            try:
                with open(session_path) as f:
                    data = json.load(f)
                sessions.append((data['timestamp'], session_path.parent))
            except Exception as e:
                print(f"Error loading session {session_path}: {e}")
                
        # Sort sessions by timestamp
        sessions.sort(reverse=True)
        
        # Update combo box
        self.session_combo.clear()
        for timestamp, path in sessions:
            self.session_combo.addItem(timestamp, path)
            
    def load_session(self, index: int):
        """Load selected session data"""
        if index < 0:
            return
            
        session_path = self.session_combo.currentData()
        if not session_path:
            return
            
        try:
            # Load session summary
            with open(session_path / "session_summary.json") as f:
                self.current_session = json.load(f)
                
            # Load performance data
            performance_file = session_path / "performance_data.h5"
            if performance_file.exists():
                with h5py.File(performance_file, 'r') as f:
                    # Load all datasets into a dictionary
                    data = {}
                    for key in f.keys():
                        data[key] = f[key][()]
                        
                    # Convert to pandas DataFrame
                    self.session_data = pd.DataFrame(data)
                    
                    # Update timeline
                    self.timeline.setMaximum(len(self.session_data) - 1)
                    self.current_frame = 0
                    
                    # Update plots with initial data
                    self.update_visualization(0)
                    
        except Exception as e:
            print(f"Error loading session: {e}")
            
    def update_visualization(self, frame: int):
        """Update visualization with data at given frame"""
        if self.session_data is None or frame >= len(self.session_data):
            return
            
        data = self.session_data.iloc[frame]
        
        # Update score plots
        timestamps = np.arange(len(self.session_data)) / 20  # Assuming 20fps
        self.total_line.setData(timestamps[:frame+1], 
                              self.session_data['total_score'][:frame+1])
        self.melody_line.setData(timestamps[:frame+1],
                               self.session_data['melody_score'][:frame+1])
        self.lyric_line.setData(timestamps[:frame+1],
                               self.session_data['lyric_score'][:frame+1])
                               
        # Update metrics plots
        for metric in self.metrics_lines:
            if metric in self.session_data:
                self.metrics_lines[metric].setData(
                    timestamps[:frame+1],
                    self.session_data[metric][:frame+1]
                )
                
        # Update feedback
        if 'feedback' in self.session_data:
            self.feedback_label.setText(data['feedback'])
            
    def update_frame(self):
        """Update current frame during playback"""
        if not self.playing or self.session_data is None:
            return
            
        self.current_frame += 1
        if self.current_frame >= len(self.session_data):
            self.playing = False
            self.play_button.setText("Play")
            return
            
        self.timeline.setValue(self.current_frame)
        self.update_visualization(self.current_frame)
        
    def toggle_playback(self):
        """Toggle playback state"""
        if self.session_data is None:
            return
            
        self.playing = not self.playing
        self.play_button.setText("Pause" if self.playing else "Play")
        
        if self.playing:
            interval = int(self.update_interval / self.playback_speed)
            self.timer.start(interval)
        else:
            self.timer.stop()
            
    def reset_playback(self):
        """Reset playback to start"""
        self.playing = False
        self.play_button.setText("Play")
        self.timer.stop()
        self.current_frame = 0
        self.timeline.setValue(0)
        self.update_visualization(0)
        
    def seek_to_position(self, position: int):
        """Seek to specific position in session"""
        self.current_frame = position
        self.update_visualization(position)
        
    def set_playback_speed(self, speed: str):
        """Set playback speed"""
        self.playback_speed = float(speed.rstrip('x'))
        if self.playing:
            interval = int(self.update_interval / self.playback_speed)
            self.timer.setInterval(interval)
            
    def export_session(self):
        """Export current session data"""
        if self.current_session is None or self.session_data is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Session",
            "",
            "CSV files (*.csv);;Excel files (*.xlsx)"
        )
        
        if not file_path:
            return
            
        try:
            if file_path.endswith('.csv'):
                self.session_data.to_csv(file_path, index=False)
            else:
                self.session_data.to_excel(file_path, index=False)
        except Exception as e:
            print(f"Error exporting session: {e}")
            
    def closeEvent(self, event):
        """Clean up when window is closed"""
        self.timer.stop()
        super().closeEvent(event)
