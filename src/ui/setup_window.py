"""
Setup window for configuring scoring session
"""

import os
import json
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
        
        # Load settings
        self.settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'setup_settings.json')
        self.settings = self._load_settings()
        
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
        
        # Restore previous settings
        self._restore_settings()
        
    def _load_settings(self) -> Dict:
        """Load settings from file
        
        Returns:
            Dict of settings, empty if file not found or error
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            logging.exception("Full traceback:")
        return {}
        
    def _save_settings(self):
        """Save current settings to file"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            
            settings = {
                'reference_file': self.reference_file,
                'input_source': self.input_combo.currentText(),
                'input_file': self.input_file
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
                
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            logging.exception("Full traceback:")
            
    def _restore_settings(self):
        """Restore settings from saved state"""
        try:
            # Restore reference file
            if 'reference_file' in self.settings and os.path.exists(self.settings['reference_file']):
                self.reference_file = self.settings['reference_file']
                self.ref_path.setText(os.path.basename(self.reference_file))
            
            # Restore input source
            if 'input_source' in self.settings:
                index = self.input_combo.findText(self.settings['input_source'])
                if index >= 0:
                    self.input_combo.setCurrentIndex(index)
            
            # Restore input file
            if 'input_file' in self.settings and os.path.exists(self.settings['input_file']):
                self.input_file = self.settings['input_file']
                self.file_path.setText(os.path.basename(self.input_file))
            
            # Update UI state
            self._on_input_changed(self.input_combo.currentText())
            self._update_start_button()
            
        except Exception as e:
            logging.error(f"Error restoring settings: {e}")
            logging.exception("Full traceback:")
        
    def _browse_reference(self):
        """Browse for reference audio file"""
        start_dir = os.path.dirname(self.reference_file) if self.reference_file else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            start_dir,
            "Audio Files (*.wav *.mp3);;All Files (*.*)"
        )
        if file_path:
            self.reference_file = file_path
            self.ref_path.setText(os.path.basename(file_path))
            self._update_start_button()
            self._save_settings()
            
    def _browse_input(self):
        """Browse for input audio file"""
        start_dir = os.path.dirname(self.input_file) if self.input_file else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Audio",
            start_dir,
            "Audio Files (*.wav *.mp3);;All Files (*.*)"
        )
        if file_path:
            self.input_file = file_path
            self.file_path.setText(os.path.basename(file_path))
            self._update_start_button()
            self._save_settings()
            
    def _on_input_changed(self, text: str):
        """Handle input source change"""
        is_file = text.lower() == "file"
        self.file_label.setVisible(is_file)
        self.file_path.setVisible(is_file)
        self.file_button.setVisible(is_file)
        if not is_file:
            self.input_file = None
        self._update_start_button()
        self._save_settings()
        
    def _update_start_button(self):
        """Update start button enabled state"""
        can_start = (
            self.reference_file is not None and
            (self.input_combo.currentText().lower() == "microphone" or
             self.input_file is not None)
        )
        self.start_button.setEnabled(can_start)
        
    def _start_session(self):
        """Start scoring session with current configuration"""
        try:
            logging.info("Starting scoring session")
            
            # Validate reference file
            if not self.reference_file:
                QMessageBox.warning(self, "Error", "Please select a reference audio file")
                return
                
            if not os.path.exists(self.reference_file):
                QMessageBox.warning(self, "Error", "Reference file does not exist")
                return
            
            # Get input source
            input_source = self.input_combo.currentText().lower()
            
            # Get input file if using file input
            input_file = None
            if input_source == 'file':
                if not self.input_file:
                    QMessageBox.warning(self, "Error", "Please select an input audio file")
                    return
                    
                if not os.path.exists(self.input_file):
                    QMessageBox.warning(self, "Error", "Input file does not exist")
                    return
                    
                input_file = self.input_file
            
            # Create session config
            config = {
                'input_source': input_source,
                'reference_file': self.reference_file,
                'input_file': input_file
            }
            
            logging.info(f"Session config: {config}")
            
            # Start session
            self.session_manager.start_session(config)
            
            # Save settings
            self._save_settings()
            
        except Exception as e:
            logging.error(f"Error starting session: {e}")
            logging.exception("Full traceback:")
            QMessageBox.critical(self, "Error", f"Failed to start session: {str(e)}")
