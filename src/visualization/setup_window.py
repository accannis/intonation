"""
Setup window for configuring audio input and reference sources
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QRadioButton, QPushButton, QFileDialog, QLabel,
                           QButtonGroup, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
import os
import json
import logging

class SetupWindow(QMainWindow):
    """Window for setting up audio input and reference sources"""
    
    # Signal emitted when setup is complete
    setup_complete = pyqtSignal(dict)  # Emits setup configuration
    
    def __init__(self, default_reference=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup Audio Sources")
        
        # Set default reference path
        if default_reference is None:
            # Default to Audio directory in project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.default_reference = os.path.join(project_root, "Audio", "healing_incarnation.wav")
        else:
            self.default_reference = default_reference
            
        # Load saved settings
        self.settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'setup_settings.json')
        self.settings = self._load_settings()
            
        # Initialize UI
        self._setup_ui()
        
        # Restore saved settings
        self._restore_settings()
        
    def _setup_ui(self):
        """Setup the UI components"""
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        
        # Input source selection
        source_group = QFrame()
        source_group.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(10)
        
        # Title for input source
        source_title = QLabel("Select Input Source:")
        source_title.setStyleSheet("font-weight: bold;")
        source_layout.addWidget(source_title)
        
        # Radio buttons for input source
        self.input_group = QButtonGroup()
        self.mic_radio = QRadioButton("Microphone")
        self.file_radio = QRadioButton("Audio File")
        self.input_group.addButton(self.mic_radio)
        self.input_group.addButton(self.file_radio)
        
        source_layout.addWidget(self.mic_radio)
        source_layout.addWidget(self.file_radio)
        
        # Input file selection
        self.input_file_label = QLabel("No input file selected")
        self.input_file_button = QPushButton("Choose Input File...")
        self.input_file_button.setEnabled(False)
        source_layout.addWidget(self.input_file_label)
        source_layout.addWidget(self.input_file_button)
        
        layout.addWidget(source_group)
        
        # Reference file selection
        ref_group = QFrame()
        ref_group.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        ref_layout = QVBoxLayout(ref_group)
        ref_layout.setSpacing(10)
        
        # Title for reference file
        ref_title = QLabel("Reference Audio:")
        ref_title.setStyleSheet("font-weight: bold;")
        ref_layout.addWidget(ref_title)
        
        # Check if default reference exists
        if os.path.exists(self.default_reference):
            ref_text = f"Current: {os.path.basename(self.default_reference)}"
        else:
            ref_text = "No reference file selected"
            self.default_reference = None
            
        self.ref_file_label = QLabel(ref_text)
        self.ref_file_button = QPushButton("Change Reference File...")
        ref_layout.addWidget(self.ref_file_label)
        ref_layout.addWidget(self.ref_file_button)
        
        layout.addWidget(ref_group)
        
        # Start session button
        self.start_button = QPushButton("Start Session")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.start_button)
        
        # Set window size
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        # Connect signals
        self.file_radio.toggled.connect(self._on_input_source_changed)
        self.input_file_button.clicked.connect(self._choose_input_file)
        self.ref_file_button.clicked.connect(self._choose_reference_file)
        self.start_button.clicked.connect(self._on_start_session)
        
        # Initialize state
        self.input_file_path = self.settings.get('input_file')
        self.ref_file_path = self.settings.get('reference_file', self.default_reference)
        
        # Update UI based on saved settings
        if self.input_file_path:
            self.input_file_label.setText(f"Selected: {os.path.basename(self.input_file_path)}")
            
        if self.ref_file_path:
            self.ref_file_label.setText(f"Current: {os.path.basename(self.ref_file_path)}")
        
    def _load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
        return {}
        
    def _save_settings(self):
        """Save settings to file"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            
            settings = {
                'input_source': 'microphone' if self.mic_radio.isChecked() else 'file',
                'input_file': self.input_file_path,
                'reference_file': self.ref_file_path
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
                
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            
    def _restore_settings(self):
        """Restore settings from saved state"""
        try:
            input_source = self.settings.get('input_source', 'microphone')
            if input_source == 'file':
                self.file_radio.setChecked(True)
            else:
                self.mic_radio.setChecked(True)
                
            self._update_start_button()
            
        except Exception as e:
            logging.error(f"Error restoring settings: {e}")
        
    def _on_input_source_changed(self, checked):
        """Handle input source radio button changes"""
        self.input_file_button.setEnabled(self.file_radio.isChecked())
        self._update_start_button()
        
    def _choose_input_file(self):
        """Open file dialog to choose input audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Audio File",
            os.path.dirname(self.input_file_path) if self.input_file_path else "",
            "Audio Files (*.wav *.mp3);;All Files (*.*)"
        )
        if file_path:
            self.input_file_path = file_path
            self.input_file_label.setText(f"Selected: {os.path.basename(file_path)}")
            self._update_start_button()
            
    def _choose_reference_file(self):
        """Open file dialog to choose reference audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio File",
            os.path.dirname(self.ref_file_path) if self.ref_file_path else "",
            "Audio Files (*.wav *.mp3);;All Files (*.*)"
        )
        if file_path:
            self.ref_file_path = file_path
            self.ref_file_label.setText(f"Current: {os.path.basename(file_path)}")
            self._update_start_button()
            
    def _update_start_button(self):
        """Update start button enabled state based on selections"""
        can_start = True
        
        # Check if we have a reference file
        if self.ref_file_path is None or not os.path.exists(self.ref_file_path):
            can_start = False
        
        # Check input source requirements
        if self.file_radio.isChecked():
            can_start = can_start and self.input_file_path is not None
            
        self.start_button.setEnabled(can_start)
        
    def _on_start_session(self):
        """Handle start session button click"""
        # Save settings before starting
        self._save_settings()
        
        # Verify reference file exists
        if not os.path.exists(self.ref_file_path):
            QMessageBox.critical(
                self,
                "Error",
                "Reference audio file not found. Please select a valid reference file.",
                QMessageBox.StandardButton.Ok
            )
            return
            
        # If using file input, verify input file exists
        if self.file_radio.isChecked() and not os.path.exists(self.input_file_path):
            QMessageBox.critical(
                self,
                "Error",
                "Input audio file not found. Please select a valid input file.",
                QMessageBox.StandardButton.Ok
            )
            return
            
        config = {
            'input_source': 'microphone' if self.mic_radio.isChecked() else 'file',
            'input_file': self.input_file_path if self.file_radio.isChecked() else None,
            'reference_file': self.ref_file_path
        }
        
        self.setup_complete.emit(config)
        self.close()
