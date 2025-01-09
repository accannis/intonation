"""
Setup window for configuring scoring session
"""

import os
import logging
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QComboBox, QPushButton, QFileDialog, QMessageBox)
from PyQt6.QtCore import pyqtSignal
from typing import Optional, Dict

class SetupWindow(QMainWindow):
    """Window for configuring scoring session"""
    
    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager
        
        self.setWindowTitle("Session Setup")
        self.setGeometry(100, 100, 600, 200)
        
        # Create central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Reference file selection
        ref_layout = QHBoxLayout()
        self.ref_label = QLabel("Reference Audio:")
        ref_layout.addWidget(self.ref_label)
        
        self.ref_path = QLabel("No file selected")
        ref_layout.addWidget(self.ref_path)
        
        self.ref_button = QPushButton("Browse...")
        self.ref_button.clicked.connect(self._browse_reference)
        ref_layout.addWidget(self.ref_button)
        
        layout.addLayout(ref_layout)
        
        # Input source selection
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input Source:")
        input_layout.addWidget(self.input_label)
        
        self.input_combo = QComboBox()
        self.input_combo.addItems(["Microphone", "File"])
        self.input_combo.currentTextChanged.connect(self._on_input_changed)
        input_layout.addWidget(self.input_combo)
        
        layout.addLayout(input_layout)
        
        # Input file selection (hidden initially)
        self.file_layout = QHBoxLayout()
        self.file_label = QLabel("Input Audio:")
        self.file_layout.addWidget(self.file_label)
        
        self.file_path = QLabel("No file selected")
        self.file_layout.addWidget(self.file_path)
        
        self.file_button = QPushButton("Browse...")
        self.file_button.clicked.connect(self._browse_input)
        self.file_layout.addWidget(self.file_button)
        
        layout.addLayout(self.file_layout)
        self.file_label.hide()
        self.file_path.hide()
        self.file_button.hide()
        
        # Start button
        self.start_button = QPushButton("Start Session")
        self.start_button.clicked.connect(self._start_session)
        self.start_button.setEnabled(False)
        layout.addWidget(self.start_button)
        
        # Store paths
        self.reference_file: Optional[str] = None
        self.input_file: Optional[str] = None
        
    def _browse_reference(self):
        """Browse for reference audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            "",
            "Audio Files (*.wav *.mp3);;All Files (*.*)"
        )
        if file_path:
            self.reference_file = file_path
            self.ref_path.setText(os.path.basename(file_path))
            self._update_start_button()
            
    def _browse_input(self):
        """Browse for input audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Audio",
            "",
            "Audio Files (*.wav *.mp3);;All Files (*.*)"
        )
        if file_path:
            self.input_file = file_path
            self.file_path.setText(os.path.basename(file_path))
            self._update_start_button()
            
    def _on_input_changed(self, text: str):
        """Handle input source change"""
        is_file = text.lower() == "file"
        self.file_label.setVisible(is_file)
        self.file_path.setVisible(is_file)
        self.file_button.setVisible(is_file)
        if not is_file:
            self.input_file = None
        self._update_start_button()
        
    def _update_start_button(self):
        """Update start button enabled state"""
        can_start = (
            self.reference_file is not None and
            (self.input_combo.currentText().lower() == "microphone" or
             self.input_file is not None)
        )
        self.start_button.setEnabled(can_start)
        
    def _start_session(self):
        """Start scoring session"""
        try:
            # Create session config
            config = {
                'reference_file': self.reference_file,
                'input_source': 'file' if self.input_combo.currentText().lower() == "file" else 'microphone'
            }
            if config['input_source'] == 'file':
                config['input_file'] = self.input_file
                
            # Start or reveal session
            self.session_manager.start_session(config)
            
        except Exception as e:
            logging.error(f"Error starting session: {e}")
            logging.exception("Full traceback:")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to start session: {str(e)}",
                QMessageBox.StandardButton.Ok
            )
