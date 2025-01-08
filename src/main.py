"""
Main application entry point for the singing scoring application
"""

import os
import sys
import logging
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

from src.visualization.setup_window import SetupWindow
from src.audio_processing.scoring_session import ScoringSession

class SingingScorer:
    def __init__(self):
        """Initialize the singing application"""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize current session
        self.current_session = None
        
        # Show setup window
        self.setup_window = SetupWindow()
        self.setup_window.setup_complete.connect(self._on_setup_complete)
        self.setup_window.show()
        
    def _cleanup_session(self):
        """Clean up current session if one exists"""
        if self.current_session is not None:
            self.current_session.cleanup()
            self.current_session = None
            
    def _on_setup_complete(self, config):
        """Handle setup completion"""
        try:
            logging.info(f"Setup complete with config: {config}")
            
            # Clean up any existing session
            self._cleanup_session()
            
            # Create new session
            self.current_session = ScoringSession(config)
            
        except Exception as e:
            logging.error(f"Error in setup completion: {e}")
            logging.exception("Full traceback:")
            QMessageBox.critical(
                None,
                "Setup Error",
                f"Error completing setup: {str(e)}",
                QMessageBox.StandardButton.Ok
            )

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    scorer = SingingScorer()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
